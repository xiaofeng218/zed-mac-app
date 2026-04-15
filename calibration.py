from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Calibration:
    K: np.ndarray
    baseline_m: float
    scale: float


def load_calibration(path: str | Path, scale: float = 1.0) -> Calibration:
    path = Path(path)
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    if not path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {path}")

    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(lines) == 2:
        values = [float(x) for x in lines[0].split()]
        if len(values) != 9:
            raise ValueError(f"Expected 9 intrinsic values, got {len(values)}")
        K = np.asarray(values, dtype=np.float32).reshape(3, 3)
        baseline_m = float(lines[1])
    elif len(lines) == 4:
        rows = []
        for i in range(3):
            values = [float(x) for x in lines[i].split()]
            if len(values) != 3:
                raise ValueError(f"Expected 3 intrinsic values on row {i + 1}, got {len(values)}")
            rows.append(values)
        K = np.asarray(rows, dtype=np.float32)
        baseline_m = float(lines[3])
    else:
        raise ValueError("Calibration file must be either 2 lines (flattened K + baseline) or 4 lines (3 rows of K + baseline).")

    K = K.copy()
    K[:2] *= scale
    return Calibration(K=K, baseline_m=baseline_m, scale=float(scale))
