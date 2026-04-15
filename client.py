from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib
import json
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np

from calibration import load_calibration
from protocol import encode_frame_packet

try:
    import cv2
except Exception:  # pragma: no cover - handled at runtime
    cv2 = None

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover - handled at runtime
    imageio = None

try:
    import websockets
except Exception:  # pragma: no cover - handled at runtime
    websockets = None


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
LOGGER = logging.getLogger(__name__)


def _candidate_right_names(left_name: str) -> list[str]:
    cands = [left_name]
    if left_name.startswith("L-"):
        cands.append("R-" + left_name[2:])
    if left_name.startswith("left_"):
        cands.append("right_" + left_name[5:])
    return cands


def match_stereo_pairs(left_names: list[str], right_names: list[str]) -> list[tuple[str, str]]:
    right_map = {name: name for name in right_names}
    pairs: list[tuple[str, str]] = []
    for left_name in left_names:
        for cand in _candidate_right_names(left_name):
            if cand in right_map:
                pairs.append((left_name, right_map[cand]))
                break
    return pairs


def list_image_names(image_dir: str | Path) -> list[str]:
    image_dir = Path(image_dir)
    return sorted(
        p.name for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def normalize_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim != 3:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    if image.shape[2] > 3:
        image = image[..., :3]
    return image.astype(np.uint8, copy=False)


def preprocess_for_upload(image_rgb: np.ndarray, scale: float = 1.0) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("cv2 is not installed in the current environment")
    image_rgb = normalize_rgb(image_rgb)
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    if scale == 1.0:
        return image_rgb
    return cv2.resize(image_rgb, fx=float(scale), fy=float(scale), dsize=None)


def encode_rgb_to_jpeg(image_rgb: np.ndarray) -> bytes:
    if cv2 is None:
        raise RuntimeError("cv2 is not installed in the current environment")
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Failed to encode RGB image as JPEG")
    return encoded.tobytes()


def build_stereo_pairs(left_dir: str | Path, right_dir: str | Path) -> list[tuple[Path, Path]]:
    left_dir = Path(left_dir)
    right_dir = Path(right_dir)
    pairs = match_stereo_pairs(list_image_names(left_dir), list_image_names(right_dir))
    return [(left_dir / left_name, right_dir / right_name) for left_name, right_name in pairs]


def build_start_stream_message(stream_name: str, intrinsic_file: str | Path, scale: float = 1.0) -> dict[str, object]:
    calibration = load_calibration(intrinsic_file, scale=scale)
    return {
        "type": "start_stream",
        "stream_name": stream_name,
        "K": calibration.K.reshape(-1).astype(float).tolist(),
        "baseline_m": float(calibration.baseline_m),
    }


def _scaled_intrinsics_from_flat_list(k_values: list[float], scale: float) -> list[float]:
    if len(k_values) != 9:
        raise ValueError(f"Expected 9 intrinsic values, got {len(k_values)}")
    scaled = [float(value) for value in k_values]
    scaled[0] *= scale
    scaled[2] *= scale
    scaled[4] *= scale
    scaled[5] *= scale
    return scaled


def build_start_stream_message_from_calibration(
    stream_name: str,
    calibration: dict[str, Any],
    scale: float = 1.0,
) -> dict[str, object]:
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    k_values = calibration.get("K")
    baseline_m = calibration.get("baseline_m")
    if not isinstance(k_values, list):
        raise ValueError("calibration must include K as a flattened list")
    if baseline_m is None:
        raise ValueError("calibration must include baseline_m")
    return {
        "type": "start_stream",
        "stream_name": stream_name,
        "K": _scaled_intrinsics_from_flat_list(k_values, scale),
        "baseline_m": float(baseline_m),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stream live ZED stereo frames to the live service")
    parser.add_argument("--ws_url", type=str, required=True)
    parser.add_argument("--stream_name", type=str, default="zed-live")
    parser.add_argument("--resolution", type=str, default="HD720")
    parser.add_argument("--camera_fps", type=int, default=30)
    parser.add_argument("--color_space", type=str, default="RGB")
    parser.add_argument("--fps", type=float, default=15.0, help="Upload FPS limit; 0 means upload every available frame")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--max_frames", type=int, default=0, help="Stop after sending this many frames; 0 means unbounded")
    return parser


async def replay_stereo_pairs(
    left_dir: str,
    right_dir: str,
    intrinsic_file: str,
    ws_url: str,
    fps: float = 15.0,
    scale: float = 1.0,
) -> None:
    if websockets is None:
        raise RuntimeError("websockets is not installed in the current environment")
    if imageio is None:
        raise RuntimeError("imageio is not installed in the current environment")
    frame_interval = 1.0 / fps if fps > 0 else 0.0
    pairs = build_stereo_pairs(left_dir, right_dir)
    async with websockets.connect(ws_url, max_size=None) as ws:
        await ws.send(json.dumps(build_start_stream_message("replay", intrinsic_file, scale=scale)))
        for frame_idx, (left_path, right_path) in enumerate(pairs):
            left_rgb = preprocess_for_upload(imageio.imread(left_path), scale=scale)
            right_rgb = preprocess_for_upload(imageio.imread(right_path), scale=scale)
            packet = encode_frame_packet(
                frame_idx=frame_idx,
                timestamp_ns=time.time_ns(),
                left_jpeg=encode_rgb_to_jpeg(left_rgb),
                right_jpeg=encode_rgb_to_jpeg(right_rgb),
            )
            await ws.send(packet)
            if frame_interval > 0:
                await asyncio.sleep(frame_interval)


async def stream_from_capture(
    capture: Any,
    websocket: Any,
    *,
    stream_name: str = "zed-live",
    scale: float = 1.0,
    fps: float = 15.0,
    max_frames: int | None = None,
    sleep_fn=asyncio.sleep,
) -> None:
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    if max_frames is not None and max_frames < 0:
        raise ValueError(f"max_frames must be non-negative, got {max_frames}")

    calibration = capture.get_calibration()
    await websocket.send(json.dumps(build_start_stream_message_from_calibration(stream_name, calibration, scale=scale)))
    LOGGER.info("start_stream sent stream_name=%s scale=%s", stream_name, scale)

    frame_interval = 1.0 / fps if fps > 0 else 0.0
    sent_frames = 0
    try:
        while max_frames is None or sent_frames < max_frames:
            frame = capture.grab()
            if frame is None:
                await sleep_fn(0.005)
                continue
            LOGGER.info(
                "frame grabbed idx=%s left_shape=%s right_shape=%s",
                frame["frame_idx"],
                np.asarray(frame["left_rgb"]).shape,
                np.asarray(frame["right_rgb"]).shape,
            )

            left_rgb = preprocess_for_upload(frame["left_rgb"], scale=scale)
            right_rgb = preprocess_for_upload(frame["right_rgb"], scale=scale)
            left_jpeg = encode_rgb_to_jpeg(left_rgb)
            right_jpeg = encode_rgb_to_jpeg(right_rgb)
            LOGGER.info(
                "frame encoded idx=%s left_bytes=%s right_bytes=%s",
                frame["frame_idx"],
                len(left_jpeg),
                len(right_jpeg),
            )
            packet = encode_frame_packet(
                frame_idx=int(frame["frame_idx"]),
                timestamp_ns=int(frame["timestamp_ns"]),
                left_jpeg=left_jpeg,
                right_jpeg=right_jpeg,
            )
            LOGGER.info("frame sending idx=%s packet_bytes=%s", frame["frame_idx"], len(packet))
            await websocket.send(packet)
            LOGGER.info("frame sent idx=%s", frame["frame_idx"])
            sent_frames += 1
            if frame_interval > 0:
                await sleep_fn(frame_interval)
    finally:
        close = getattr(capture, "close", None)
        if callable(close):
            with contextlib.suppress(Exception):
                close()


def load_capture_factory(module_name: str = "zed_native") -> type[Any]:
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        raise RuntimeError(f"Native capture module '{module_name}' could not be imported") from None
    capture_factory = getattr(module, "ZedCapture", None)
    if capture_factory is None:
        raise RuntimeError(f"Native capture module '{module_name}' does not export ZedCapture")
    return capture_factory


def open_live_capture(
    *,
    resolution: str,
    camera_fps: int,
    color_space: str,
    capture_factory: type[Any] | None = None,
) -> Any:
    if capture_factory is None:
        capture_factory = load_capture_factory()
    capture = capture_factory()
    capture.open(resolution=resolution, fps=int(camera_fps), color_space=color_space)
    return capture


async def stream_live_capture(
    ws_url: str,
    *,
    stream_name: str,
    resolution: str,
    camera_fps: int,
    color_space: str,
    fps: float,
    scale: float,
    max_frames: int | None = None,
    capture_factory: type[Any] | None = None,
) -> None:
    if websockets is None:
        raise RuntimeError("websockets is not installed in the current environment")
    capture = open_live_capture(
        resolution=resolution,
        camera_fps=camera_fps,
        color_space=color_space,
        capture_factory=capture_factory,
    )
    async with websockets.connect(ws_url, max_size=None) as ws:
        await stream_from_capture(
            capture,
            ws,
            stream_name=stream_name,
            scale=scale,
            fps=fps,
            max_frames=max_frames,
        )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    args = build_arg_parser().parse_args()
    asyncio.run(
        stream_live_capture(
            ws_url=args.ws_url,
            stream_name=args.stream_name,
            resolution=args.resolution,
            camera_fps=args.camera_fps,
            color_space=args.color_space,
            fps=args.fps,
            scale=args.scale,
            max_frames=None if args.max_frames == 0 else args.max_frames,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
