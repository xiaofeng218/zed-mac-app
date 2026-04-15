from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import numpy as np

try:
    import rerun as rr
except Exception:  # pragma: no cover - handled at runtime
    rr = None


@dataclass(frozen=True)
class EntityPaths:
    left: str
    right: str
    disparity: str
    depth: str
    points: str
    latency_ms: str
    input_fps: str
    dropped_frames: str


@dataclass(frozen=True)
class RemoteViewerUrls:
    web_viewer: str
    grpc_proxy: str


def build_entity_paths(session_id: str) -> EntityPaths:
    prefix = f"/session/{session_id}"
    return EntityPaths(
        left=f"{prefix}/input/left",
        right=f"{prefix}/input/right",
        disparity=f"{prefix}/output/disparity",
        depth=f"{prefix}/output/depth",
        points=f"{prefix}/output/points",
        latency_ms=f"{prefix}/metrics/latency_ms",
        input_fps=f"{prefix}/metrics/input_fps",
        dropped_frames=f"{prefix}/metrics/dropped_frames",
    )


def build_remote_viewer_urls(public_host: str, web_port: int, grpc_port: int) -> RemoteViewerUrls:
    return RemoteViewerUrls(
        web_viewer=f"http://{public_host}:{int(web_port)}",
        grpc_proxy=f"rerun+http://{public_host}:{int(grpc_port)}/proxy",
    )


class RerunStreamLogger:
    def __init__(
        self,
        recording_name: str = "stereo_depth_live",
        spawn: bool = True,
        *,
        mode: str = "spawn",
        web_port: int = 9090,
        grpc_port: int = 9876,
        public_host: str = "127.0.0.1",
        server_memory_limit: str = "25%",
    ):
        if rr is None:
            raise RuntimeError("rerun is not installed in the current environment")
        self.recording_name = recording_name
        self.spawn = bool(spawn)
        self.mode = mode
        self.web_port = int(web_port)
        self.grpc_port = int(grpc_port)
        self.public_host = str(public_host)
        self.server_memory_limit = str(server_memory_limit)
        self.remote_urls: RemoteViewerUrls | None = None

        rr.init(recording_name, spawn=False)

        if self.mode == "spawn":
            self._spawn_viewer()
        elif self.mode == "web":
            self._serve_remote_viewer()
        elif self.mode != "headless":
            raise ValueError(f"Unsupported rerun mode: {self.mode}")

    def _spawn_viewer(self) -> None:
        if hasattr(rr, "spawn"):
            rr.spawn()
            return
        rr.init(self.recording_name, spawn=True)

    def _serve_remote_viewer(self) -> None:
        if not hasattr(rr, "serve_grpc"):
            raise RuntimeError("Installed rerun package does not support serve_grpc for remote viewing")

        server_uri = rr.serve_grpc(
            grpc_port=self.grpc_port,
            server_memory_limit=self.server_memory_limit,
            newest_first=False,
        )

        if hasattr(rr, "serve_web_viewer"):
            try:
                # Rerun only wires `connect_to` into the hosted page when `open_browser=True`.
                # On a headless server this may fail to launch a browser, but the web endpoint still works.
                rr.serve_web_viewer(
                    web_port=self.web_port,
                    open_browser=True,
                    connect_to=server_uri,
                )
            except Exception:
                logging.exception("Failed while opening local browser for Rerun web viewer; continuing in remote web mode")
        elif hasattr(rr, "serve_web"):
            rr.serve_web(
                open_browser=True,
                web_port=self.web_port,
                grpc_port=self.grpc_port,
                server_memory_limit=self.server_memory_limit,
            )
        else:
            raise RuntimeError("Installed rerun package does not support serve_web_viewer or serve_web")

        self.remote_urls = build_remote_viewer_urls(
            public_host=self.public_host,
            web_port=self.web_port,
            grpc_port=self.grpc_port,
        )

    def log_frame_result(self, result: dict[str, Any]) -> None:
        session_id = str(result["session_id"])
        paths = build_entity_paths(session_id)
        rr.set_time("frame_idx", sequence=int(result["frame_idx"]))
        rr.set_time("sensor_time", timestamp=np.datetime64(int(result["timestamp_ns"]), "ns"))
        rr.log(paths.left, rr.Image(result["left_rgb"]))
        rr.log(paths.right, rr.Image(result["right_rgb"]))
        rr.log(paths.disparity, rr.Image(result["disp_vis"]))
        rr.log(paths.depth, rr.DepthImage(result["depth_m"], meter=1.0))
        rr.log(paths.points, rr.Points3D(result["points_xyz"], colors=result["points_rgb"]))
        rr.log(paths.latency_ms, rr.Scalars([float(result["latency_ms"])]))
