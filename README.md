# stereo-app

Real-time stereo streaming app for ZED cameras.

- `app.py`: server entrypoint for stereo inference and Rerun visualization
- `client.py`: client entrypoint for live capture and frame upload

## Environment

- macOS for the client-side native capture build
- Python 3.10+
- CMake 3.20+
- Xcode Command Line Tools
- `libcurl`
- A supported ZED camera for live client capture
- PyTorch-compatible model weights for the server

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Build

Build the client native bridge:

```bash
cmake -S native -B native/build
cmake --build native/build --parallel
```

## Run

Start the server:

```bash
python app.py \
  --model_dir /path/to/model_best_bp2_serialize.pth \
  --host 0.0.0.0 \
  --port 8765 \
  --rr-mode web \
  --rr-web-port 9090 \
  --rr-grpc-port 9876
```

Start the client:

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

## Two-Machine Setup

Clone this repository on both the server machine and the client machine.

Server machine:

1. Install Python dependencies with `pip install -r requirements.txt`.
2. Prepare the model weights used by `app.py`.
3. Start the server and listen on port `8765`.

```bash
python app.py \
  --model_dir /path/to/model_best_bp2_serialize.pth \
  --host 0.0.0.0 \
  --port 8765 \
  --rr-mode web
```

Client machine:

1. Install Python dependencies with `pip install -r requirements.txt`.
2. Build the native bridge with:

```bash
cmake -S native -B native/build
cmake --build native/build --parallel
```

3. Connect `client.py` to the server on port `8765`.

If you access the server through SSH, forward local port `8765` to the server's port `8765`:

```sshconfig
Host zed-server
  HostName your_server_ip
  User your_username
  LocalForward 8765 localhost:8765
```

Then run the client locally with:

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

- `third_party/zed-open-capture-mac/` is vendored in this repository, so the client native build does not depend on an external clone.
- Server inference still depends on the model code serialized with the weight file loaded by `torch.load(...)`. download from [here](https://drive.google.com/drive/folders/1HuTt7UIp7gQsMiDvJwVuWmKpvFzIIMap?usp=drive_link)
