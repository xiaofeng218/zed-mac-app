from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np


class _ZedFrameBuffer(ctypes.Structure):
    _fields_ = [
        ("frame_idx", ctypes.c_uint64),
        ("timestamp_ns", ctypes.c_uint64),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("channels", ctypes.c_int),
        ("left_data", ctypes.POINTER(ctypes.c_uint8)),
        ("left_size", ctypes.c_size_t),
        ("right_data", ctypes.POINTER(ctypes.c_uint8)),
        ("right_size", ctypes.c_size_t),
    ]


def _default_library_candidates() -> list[Path]:
    base_dir = Path(__file__).resolve().parent / "native"
    return [
        Path(os.environ["STEREO_APP_ZED_NATIVE_LIB"]).expanduser() if "STEREO_APP_ZED_NATIVE_LIB" in os.environ else None,
        base_dir / "build" / "libzed_capture_bridge.dylib",
        base_dir / "build" / "libzed_capture_bridge.so",
    ]


def _resolve_library_path() -> Path:
    for candidate in _default_library_candidates():
        if candidate is not None and candidate.is_file():
            return candidate
    searched = [str(path) for path in _default_library_candidates() if path is not None]
    raise FileNotFoundError(
        "Unable to find the native ZED bridge library. "
        f"Searched: {searched}. "
        "Build native/ first or set STEREO_APP_ZED_NATIVE_LIB."
    )


class _NativeLibrary:
    def __init__(self, path: str | Path | None = None) -> None:
        library_path = Path(path) if path is not None else _resolve_library_path()
        self.lib = ctypes.CDLL(str(library_path))
        self.lib.zed_capture_create.restype = ctypes.c_void_p
        self.lib.zed_capture_destroy.argtypes = [ctypes.c_void_p]
        self.lib.zed_capture_open.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self.lib.zed_capture_open.restype = ctypes.c_int
        self.lib.zed_capture_close.argtypes = [ctypes.c_void_p]
        self.lib.zed_capture_get_calibration.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self.lib.zed_capture_get_calibration.restype = ctypes.c_int
        self.lib.zed_capture_grab.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_ZedFrameBuffer),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self.lib.zed_capture_grab.restype = ctypes.c_int
        self.lib.zed_capture_release_frame.argtypes = [ctypes.POINTER(_ZedFrameBuffer)]

    def _error_buffer(self) -> ctypes.Array[ctypes.c_char]:
        return ctypes.create_string_buffer(512)

    def _raise_if_failed(self, status: int, error_buffer: ctypes.Array[ctypes.c_char], default_message: str) -> None:
        if status == 0:
            return
        message = error_buffer.value.decode("utf-8", errors="replace") or default_message
        raise RuntimeError(message)


class ZedCapture:
    def __init__(self, library_path: str | Path | None = None) -> None:
        self._native = _NativeLibrary(library_path)
        self._handle = ctypes.c_void_p(self._native.lib.zed_capture_create())
        if not self._handle:
            raise RuntimeError("Failed to create native ZED capture handle")

    def open(self, resolution: str = "HD720", fps: int = 30, color_space: str = "RGB") -> None:
        error = self._native._error_buffer()
        status = self._native.lib.zed_capture_open(
            self._handle,
            resolution.encode("utf-8"),
            int(fps),
            color_space.encode("utf-8"),
            error,
            len(error),
        )
        self._native._raise_if_failed(status, error, "Failed to open ZED capture")

    def get_calibration(self) -> dict[str, object]:
        error = self._native._error_buffer()
        k_values = (ctypes.c_float * 9)()
        baseline_m = ctypes.c_float()
        width = ctypes.c_int()
        height = ctypes.c_int()
        status = self._native.lib.zed_capture_get_calibration(
            self._handle,
            k_values,
            ctypes.byref(baseline_m),
            ctypes.byref(width),
            ctypes.byref(height),
            error,
            len(error),
        )
        self._native._raise_if_failed(status, error, "Failed to read ZED calibration")
        return {
            "K": [float(value) for value in k_values],
            "baseline_m": float(baseline_m.value),
            "width": int(width.value),
            "height": int(height.value),
        }

    def grab(self) -> dict[str, object] | None:
        error = self._native._error_buffer()
        frame = _ZedFrameBuffer()
        status = self._native.lib.zed_capture_grab(
            self._handle,
            ctypes.byref(frame),
            error,
            len(error),
        )
        if status < 0:
            self._native._raise_if_failed(status, error, "Failed to grab ZED frame")
        if status == 0:
            return None

        try:
            left_bytes = ctypes.string_at(frame.left_data, frame.left_size)
            right_bytes = ctypes.string_at(frame.right_data, frame.right_size)
        finally:
            self._native.lib.zed_capture_release_frame(ctypes.byref(frame))

        shape = (frame.height, frame.width, frame.channels)
        left_rgb = np.frombuffer(left_bytes, dtype=np.uint8).reshape(shape).copy()
        right_rgb = np.frombuffer(right_bytes, dtype=np.uint8).reshape(shape).copy()
        return {
            "frame_idx": int(frame.frame_idx),
            "timestamp_ns": int(frame.timestamp_ns),
            "left_rgb": left_rgb,
            "right_rgb": right_rgb,
        }

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._native.lib.zed_capture_close(self._handle)
            self._native.lib.zed_capture_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass
