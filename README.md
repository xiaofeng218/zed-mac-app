# stereo-app

Real-time stereo streaming app with two entrypoints:

- `app.py`: server-side reception, inference, and Rerun visualization
- `client.py`: client-side ZED capture and frame upload

## What It Does

The client captures left/right stereo frames from a ZED camera, sends JPEG-compressed pairs plus `K + baseline` over WebSocket, and the server runs disparity, depth, and point cloud inference before publishing results to Rerun.

## Repository Layout

- `app.py`: server entrypoint
- `client.py`: client entrypoint
- `server.py`, `pipeline.py`, `rerun_logger.py`: server internals
- `calibration.py`, `zed_native.py`: client internals
- `protocol.py`: shared binary packet protocol
- `native/`: local bridge library that exposes ZED capture to Python
- `third_party/zed-open-capture-mac/`: vendored camera capture source used by the native bridge

## Requirements

- macOS with a supported ZED camera for the client side
- Python 3.10+
- CMake 3.20+
- Xcode command line tools
- `libcurl`

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Build The Client Native Library

```bash
cmake -S native -B native/build
cmake --build native/build --parallel
```

This produces `native/build/libzed_capture_bridge.dylib`, which `client.py` loads automatically.

## Run The Server

```bash
python app.py \
  --model_dir /path/to/model_best_bp2_serialize.pth \
  --host 0.0.0.0 \
  --port 8765 \
  --rr-mode web \
  --rr-web-port 9090 \
  --rr-grpc-port 9876
```

Common options:

- `--queue_size`: latest-frame queue length, usually `1`
- `--pc_stride`: point cloud downsample stride
- `--rr-mode`: `spawn`, `web`, or `headless`

## Run The Client

```bash
python client.py \
  --ws_url ws://127.0.0.1:8765 \
  --stream_name zed2i \
  --resolution HD720 \
  --camera_fps 30 \
  --color_space RGB \
  --fps 15 \
  --scale 1.0
```

## Notes

- The native client build is self-contained inside this repository because `zed-open-capture-mac` is vendored under `third_party/`.
- Server inference expects the model weights plus the model code that is serialized with them, but the local helper utilities previously imported from external `Utils` and `core.utils.utils` modules are now included in this repository.
- Do not commit `native/build/`, model weights, or local cache files.
