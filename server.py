from __future__ import annotations

"""
Server-side receiver for stereo frame uploads.

Typical local workflow:

1. Build the native bridge under `stereo_app/native`
2. Start your server process that uses `StereoStreamServer`
3. Start the uploader with:

   `python3 client.py --ws_url ws://127.0.0.1:8765`
"""

import argparse
import asyncio
from collections import deque
import contextlib
from dataclasses import dataclass, field
import json
import logging
from typing import Any
import uuid

import numpy as np

from calibration import Calibration
from protocol import decode_frame_packet

try:
    import cv2
except Exception:  # pragma: no cover - handled at runtime
    cv2 = None

try:
    import websockets
except Exception:  # pragma: no cover - handled at runtime
    websockets = None


class SessionBuffer:
    def __init__(self, max_queue_size: int = 1):
        if max_queue_size <= 0:
            raise ValueError(f"max_queue_size must be positive, got {max_queue_size}")
        self.max_queue_size = max_queue_size
        self._items: deque[Any] = deque(maxlen=max_queue_size)

    def push(self, item: Any) -> None:
        self._items.append(item)

    def pop_latest(self) -> Any | None:
        if not self._items:
            return None
        item = self._items[-1]
        self._items.clear()
        return item

    def __len__(self) -> int:
        return len(self._items)


@dataclass
class SessionState:
    session_id: str
    buffer: SessionBuffer
    accepted_frames: int = 0
    dropped_frames: int = 0
    last_error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    calibration: Calibration | None = None


def add_server_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--queue_size", type=int, default=1)
    return parser


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Receive uploaded stereo frames and run the live stereo pipeline")
    add_server_args(parser)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--valid_iters", type=int, default=8)
    parser.add_argument("--max_disp", type=int, default=192)
    parser.add_argument("--hiera", type=int, default=0)
    parser.add_argument("--remove_invisible", type=int, default=1)
    parser.add_argument("--zfar", type=float, default=100.0)
    parser.add_argument("--pc_stride", type=int, default=2)
    parser.add_argument("--recording_name", type=str, default="stereo_depth_live")
    parser.add_argument("--spawn", dest="spawn", action="store_true", default=True)
    parser.add_argument("--no_spawn", dest="spawn", action="store_false")
    return parser


def decode_jpeg_rgb(data: bytes) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("cv2 is not installed in the current environment")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode JPEG image")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


@dataclass
class StereoFrame:
    frame_idx: int
    timestamp_ns: int
    left_rgb: np.ndarray
    right_rgb: np.ndarray


class StereoStreamServer:
    def __init__(self, pipeline, rerun_logger, host: str, port: int, queue_size: int = 1):
        self.pipeline = pipeline
        self.rerun_logger = rerun_logger
        self.host = host
        self.port = port
        self.queue_size = queue_size
        self._server = None
        self._sessions: dict[str, SessionState] = {}

    async def start(self) -> None:
        if websockets is None:
            raise RuntimeError("websockets is not installed in the current environment")
        self._server = await websockets.serve(self._handle_connection, self.host, self.port, max_size=None)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def run_forever(self) -> None:
        await self.start()
        try:
            await asyncio.Future()
        finally:
            await self.stop()

    async def _handle_connection(self, websocket) -> None:
        session_id = uuid.uuid4().hex[:8]
        state = SessionState(session_id=session_id, buffer=SessionBuffer(self.queue_size))
        self._sessions[session_id] = state
        processor = asyncio.create_task(self._process_session(state))
        try:
            async for message in websocket:
                if isinstance(message, str):
                    await self._handle_control_message(websocket, state, message)
                    continue

                frame = self._decode_frame(message)
                if len(state.buffer) >= state.buffer.max_queue_size:
                    state.dropped_frames += 1
                state.buffer.push(frame)
                state.accepted_frames += 1
        finally:
            processor.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await processor
            self._sessions.pop(session_id, None)

    async def _handle_control_message(self, websocket, state: SessionState, payload: str) -> None:
        try:
            msg = json.loads(payload)
        except json.JSONDecodeError:
            await websocket.send(json.dumps({"type": "error", "message": "invalid json"}))
            return

        msg_type = msg.get("type")
        if msg_type == "start_stream":
            state.metadata["stream_name"] = msg.get("stream_name")
            state.calibration = self._parse_calibration(msg)
            await websocket.send(json.dumps({"type": "ack", "session_id": state.session_id}))
        elif msg_type == "ping":
            await websocket.send(json.dumps({"type": "pong"}))

    @staticmethod
    def _parse_calibration(msg: dict[str, Any]) -> Calibration:
        k_values = msg.get("K")
        baseline_m = msg.get("baseline_m")
        if not isinstance(k_values, list) or len(k_values) != 9:
            raise ValueError("start_stream must include K as a flattened 3x3 list")
        if baseline_m is None:
            raise ValueError("start_stream must include baseline_m")
        K = np.asarray(k_values, dtype=np.float32).reshape(3, 3)
        return Calibration(K=K, baseline_m=float(baseline_m), scale=1.0)

    def _decode_frame(self, packet: bytes) -> StereoFrame:
        decoded = decode_frame_packet(packet)
        left_rgb = decode_jpeg_rgb(decoded.left_jpeg)
        right_rgb = decode_jpeg_rgb(decoded.right_jpeg)
        return StereoFrame(
            frame_idx=decoded.frame_idx,
            timestamp_ns=decoded.timestamp_ns,
            left_rgb=left_rgb,
            right_rgb=right_rgb,
        )

    async def _process_session(self, state: SessionState) -> None:
        while True:
            frame = state.buffer.pop_latest()
            if frame is None:
                await asyncio.sleep(0.005)
                continue
            try:
                if state.calibration is None:
                    raise RuntimeError("Session calibration is not initialized")
                print(f"processing frame={frame.frame_idx} left={frame.left_rgb.shape} right={frame.right_rgb.shape}")
                result = self.pipeline.process_frame(
                    frame.left_rgb,
                    frame.right_rgb,
                    frame.frame_idx,
                    frame.timestamp_ns,
                    state.session_id,
                    calibration=state.calibration,
                )
                self.rerun_logger.log_frame_result(result)
            except Exception as exc:  # pragma: no cover - runtime safety path
                state.last_error = str(exc)
                logging.exception("Failed to process session %s frame %s", state.session_id, frame.frame_idx)


def _default_pipeline_factory(
    model_path: str,
    *,
    calibration,
    scale: float,
    valid_iters: int,
    max_disp: int,
    hiera: int,
    remove_invisible: int,
    zfar: float,
    pc_stride: int,
):
    from pipeline import StereoInferencePipeline

    return StereoInferencePipeline.from_model_path(
        model_path,
        calibration=calibration,
        scale=scale,
        valid_iters=valid_iters,
        max_disp=max_disp,
        hiera=hiera,
        remove_invisible=remove_invisible,
        zfar=zfar,
        pc_stride=pc_stride,
    )


def _default_logger_factory(*, recording_name: str, spawn: bool):
    from rerun_logger import RerunStreamLogger

    return RerunStreamLogger(recording_name=recording_name, spawn=spawn)


def create_runtime_server(
    args,
    *,
    pipeline_factory=None,
    logger_factory=None,
    server_factory=None,
):
    pipeline_factory = pipeline_factory or _default_pipeline_factory
    logger_factory = logger_factory or _default_logger_factory
    server_factory = server_factory or StereoStreamServer

    pipeline = pipeline_factory(
        args.model_path,
        calibration=None,
        scale=args.scale,
        valid_iters=args.valid_iters,
        max_disp=args.max_disp,
        hiera=args.hiera,
        remove_invisible=args.remove_invisible,
        zfar=args.zfar,
        pc_stride=args.pc_stride,
    )
    rerun_logger = logger_factory(recording_name=args.recording_name, spawn=args.spawn)
    return server_factory(
        pipeline,
        rerun_logger,
        host=args.host,
        port=args.port,
        queue_size=args.queue_size,
    )


def main() -> int:
    args = build_arg_parser().parse_args()
    server = create_runtime_server(args)
    asyncio.run(server.run_forever())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
