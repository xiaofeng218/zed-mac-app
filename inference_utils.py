from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F


AMP_DTYPE = torch.float16


class InputPadder:
    """Pads images so the model input shape is divisible by a fixed stride."""

    def __init__(self, dims, mode: str = "sintel", divis_by: int = 8, force_square: bool = False):
        self.ht, self.wd = dims[-2:]
        if force_square:
            max_side = max(self.ht, self.wd)
            pad_ht = ((max_side // divis_by) + 1) * divis_by - self.ht
            pad_wd = ((max_side // divis_by) + 1) * divis_by - self.wd
        else:
            pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
            pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by

        if mode == "sintel":
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        assert all(x.ndim == 4 for x in inputs)
        return [F.pad(x, self._pad, mode="replicate") for x in inputs]

    def unpad(self, x):
        assert x.ndim == 4
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0] : c[1], c[2] : c[3]]


def depth2xyzmap(depth: np.ndarray, K, uvs: np.ndarray | None = None, zmin: float = 0.1) -> np.ndarray:
    invalid_mask = depth < zmin
    height, width = depth.shape[:2]
    if uvs is None:
        vs, us = np.meshgrid(np.arange(0, height), np.arange(0, width), sparse=False, indexing="ij")
        vs = vs.reshape(-1)
        us = us.reshape(-1)
    else:
        uvs = uvs.round().astype(int)
        us = uvs[:, 0]
        vs = uvs[:, 1]

    zs = depth[vs, us]
    xs = (us - K[0, 2]) * zs / K[0, 0]
    ys = (vs - K[1, 2]) * zs / K[1, 1]
    pts = np.stack((xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)), 1)
    xyz_map = np.zeros((height, width, 3), dtype=np.float32)
    xyz_map[vs, us] = pts
    if invalid_mask.any():
        xyz_map[invalid_mask] = 0
    return xyz_map


def vis_disparity(
    disp: np.ndarray,
    min_val=None,
    max_val=None,
    invalid_thres=np.inf,
    color_map=cv2.COLORMAP_TURBO,
    cmap=None,
    other_output=None,
) -> np.ndarray:
    disp = disp.copy()
    height, width = disp.shape[:2]
    invalid_mask = disp >= invalid_thres
    if other_output is None:
        other_output = {}
    if (invalid_mask == 0).sum() == 0:
        other_output["min_val"] = None
        other_output["max_val"] = None
        return np.zeros((height, width, 3))
    if min_val is None:
        min_val = disp[invalid_mask == 0].min()
    if max_val is None:
        max_val = disp[invalid_mask == 0].max()
    other_output["min_val"] = min_val
    other_output["max_val"] = max_val
    vis = ((disp - min_val) / (max_val - min_val)).clip(0, 1) * 255
    if cmap is None:
        vis = cv2.applyColorMap(vis.clip(0, 255).astype(np.uint8), color_map)[..., ::-1]
    else:
        vis = cmap(vis.astype(np.uint8))[..., :3] * 255
    if invalid_mask.any():
        vis[invalid_mask] = 0
    return vis.astype(np.uint8)
