from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import os
import time

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - handled at runtime
    torch = None

try:
    from inference_utils import AMP_DTYPE, InputPadder, depth2xyzmap, vis_disparity
except Exception:  # pragma: no cover - handled at runtime
    AMP_DTYPE = None
    InputPadder = None
    depth2xyzmap = None
    vis_disparity = None


def _require_model_runtime() -> None:
    missing: list[str] = []
    if torch is None:
        missing.append("torch")
    if AMP_DTYPE is None or depth2xyzmap is None or vis_disparity is None or InputPadder is None:
        missing.append("local inference helpers")
    if missing:
        raise RuntimeError(
            "Server-side inference dependencies are unavailable: "
            + ", ".join(missing)
            + ". Install the model runtime before launching app.py."
        )


def disparity_to_depth(disp: np.ndarray, fx: float, baseline_m: float) -> np.ndarray:
    if fx <= 0:
        raise ValueError(f"fx must be positive, got {fx}")
    if baseline_m <= 0:
        raise ValueError(f"baseline_m must be positive, got {baseline_m}")

    disp = np.asarray(disp, dtype=np.float32)
    depth = np.full_like(disp, np.inf, dtype=np.float32)
    valid = disp > 0
    depth[valid] = (fx * baseline_m) / disp[valid]
    return depth


def depth_to_points(
    depth: np.ndarray,
    image_rgb: np.ndarray,
    K: np.ndarray,
    zfar: float,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if depth.shape[:2] != image_rgb.shape[:2]:
        raise ValueError(f"depth/image shape mismatch: {depth.shape} vs {image_rgb.shape}")

    xyz_map = depth2xyzmap(np.asarray(depth, dtype=np.float32), np.asarray(K, dtype=np.float32))
    xyz_map = xyz_map[::stride, ::stride]
    colors = image_rgb[::stride, ::stride]

    points = xyz_map.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    valid = np.isfinite(points[:, 2]) & (points[:, 2] > 0) & (points[:, 2] <= zfar)
    return points[valid].astype(np.float32), colors[valid]


@dataclass
class StereoInferencePipeline:
    model: object | None = None
    calibration: object | None = None
    scale: float = 1.0
    valid_iters: int = 8
    max_disp: int = 192
    hiera: int = 0
    remove_invisible: int = 1
    zfar: float = 100.0
    pc_stride: int = 2
    device: str | None = None

    def __post_init__(self) -> None:
        if self.device is None:
            cuda_available = bool(torch is not None and torch.cuda.is_available())
            self.device = "cuda" if cuda_available else "cpu"

    def validate_pair(self, left_rgb: np.ndarray, right_rgb: np.ndarray) -> None:
        if left_rgb.ndim != 3 or right_rgb.ndim != 3:
            raise ValueError("left_rgb and right_rgb must be HxWxC arrays")
        if left_rgb.shape[:2] != right_rgb.shape[:2]:
            raise ValueError(f"left/right shape mismatch: {left_rgb.shape} vs {right_rgb.shape}")
        if left_rgb.shape[2] < 3 or right_rgb.shape[2] < 3:
            raise ValueError("left_rgb and right_rgb must have at least 3 channels")

    @staticmethod
    def normalize_rgb(image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        if image.ndim != 3:
            raise ValueError(f"Unsupported image shape: {image.shape}")
        if image.shape[2] > 3:
            image = image[..., :3]
        return image.astype(np.uint8, copy=False)

    @classmethod
    def from_model_path(
        cls,
        model_dir: str,
        calibration,
        *,
        scale: float = 1.0,
        valid_iters: int = 8,
        max_disp: int = 192,
        hiera: int = 0,
        remove_invisible: int = 1,
        zfar: float = 100.0,
        pc_stride: int = 2,
    ) -> "StereoInferencePipeline":
        _require_model_runtime()
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(model_dir)), "cfg.yaml")
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"Model config not found next to weight file: {cfg_path}")

        model = torch.load(model_dir, map_location="cpu", weights_only=False)
        model.args.valid_iters = valid_iters
        model.args.max_disp = max_disp
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        return cls(
            model=model,
            calibration=calibration,
            scale=scale,
            valid_iters=valid_iters,
            max_disp=max_disp,
            hiera=hiera,
            remove_invisible=remove_invisible,
            zfar=zfar,
            pc_stride=pc_stride,
        )

    def process_frame(
        self,
        left_rgb: np.ndarray,
        right_rgb: np.ndarray,
        frame_idx: int,
        timestamp_ns: int,
        session_id: str,
        calibration=None,
    ) -> dict[str, object]:
        _require_model_runtime()
        if self.model is None:
            raise RuntimeError("StereoInferencePipeline.model is not initialized")
        calibration = calibration or self.calibration
        if calibration is None:
            raise RuntimeError("StereoInferencePipeline.calibration is not initialized")

        left_rgb = self.normalize_rgb(left_rgb)
        right_rgb = self.normalize_rgb(right_rgb)
        self.validate_pair(left_rgb, right_rgb)

        h, w = left_rgb.shape[:2]
        left_vis = left_rgb.copy()
        right_vis = right_rgb.copy()

        t0 = torch.as_tensor(left_rgb, device=self.device).float()[None].permute(0, 3, 1, 2)
        t1 = torch.as_tensor(right_rgb, device=self.device).float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(t0.shape, divis_by=32, force_square=False)
        t0, t1 = padder.pad(t0, t1)

        if self.device == "cuda":
            autocast_ctx = torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE)
            torch.cuda.synchronize()
        else:
            autocast_ctx = nullcontext()

        start = time.perf_counter()
        with autocast_ctx:
            if not self.hiera:
                disp = self.model.forward(t0, t1, iters=self.valid_iters, test_mode=True, optimize_build_volume="pytorch1")
            else:
                disp = self.model.run_hierachical(t0, t1, iters=self.valid_iters, test_mode=True, small_ratio=0.5)
        if self.device == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start) * 1000.0

        disp = padder.unpad(disp.float())
        disp = disp.detach().cpu().numpy().reshape(h, w).clip(0, None)
        disp_vis = vis_disparity(disp, min_val=None, max_val=None)

        if self.remove_invisible:
            yy, xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing="ij")
            invalid = (xx - disp) < 0
            disp = disp.copy()
            disp[invalid] = np.inf

        depth_m = disparity_to_depth(disp, fx=float(calibration.K[0, 0]), baseline_m=float(calibration.baseline_m))
        points_xyz, points_rgb = depth_to_points(
            depth_m,
            left_vis,
            calibration.K,
            zfar=self.zfar,
            stride=self.pc_stride,
        )

        return {
            "session_id": session_id,
            "frame_idx": int(frame_idx),
            "timestamp_ns": int(timestamp_ns),
            "left_rgb": left_vis,
            "right_rgb": right_vis,
            "disp": disp,
            "disp_vis": disp_vis,
            "depth_m": depth_m,
            "points_xyz": points_xyz,
            "points_rgb": points_rgb,
            "latency_ms": float(latency_ms),
        }
