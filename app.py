from __future__ import annotations

import argparse
import asyncio

from pipeline import StereoInferencePipeline
from rerun_logger import RerunStreamLogger
from server import StereoStreamServer, add_server_args


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live stereo depth streaming service")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--valid_iters", type=int, default=8)
    parser.add_argument("--max_disp", type=int, default=192)
    parser.add_argument("--hiera", type=int, default=0)
    parser.add_argument("--remove_invisible", type=int, default=1)
    parser.add_argument("--zfar", type=float, default=100.0)
    parser.add_argument("--pc_stride", type=int, default=2)
    parser.add_argument("--rr-recording-name", type=str, default="stereo_depth_live", dest="rr_recording_name")
    parser.add_argument("--rr-mode", type=str, choices=("spawn", "web", "headless"), default="spawn", dest="rr_mode")
    parser.add_argument("--rr-web-port", type=int, default=9090, dest="rr_web_port")
    parser.add_argument("--rr-grpc-port", type=int, default=9876, dest="rr_grpc_port")
    parser.add_argument("--rr-public-host", type=str, default="127.0.0.1", dest="rr_public_host")
    parser.add_argument("--rr-memory-limit", type=str, default="25%", dest="rr_memory_limit")
    add_server_args(parser)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    pipeline = StereoInferencePipeline.from_model_path(
        args.model_dir,
        None,
        scale=1.0,
        valid_iters=args.valid_iters,
        max_disp=args.max_disp,
        hiera=args.hiera,
        remove_invisible=args.remove_invisible,
        zfar=args.zfar,
        pc_stride=args.pc_stride,
    )
    rerun_logger = RerunStreamLogger(
        recording_name=args.rr_recording_name,
        spawn=(args.rr_mode == "spawn"),
        mode=args.rr_mode,
        web_port=args.rr_web_port,
        grpc_port=args.rr_grpc_port,
        public_host=args.rr_public_host,
        server_memory_limit=args.rr_memory_limit,
    )
    if rerun_logger.remote_urls is not None:
        print(f"Rerun web viewer: {rerun_logger.remote_urls.web_viewer}")
        print(f"Rerun native viewer: {rerun_logger.remote_urls.grpc_proxy}")
    server = StereoStreamServer(
        pipeline=pipeline,
        rerun_logger=rerun_logger,
        host=args.host,
        port=args.port,
        queue_size=args.queue_size,
    )
    asyncio.run(server.run_forever())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
