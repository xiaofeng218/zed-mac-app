from __future__ import annotations

import struct
from dataclasses import dataclass


MAGIC = b"FFS1"
VERSION = 1
HEADER_STRUCT = struct.Struct(">4sBQQII")


@dataclass(frozen=True)
class FramePacket:
    frame_idx: int
    timestamp_ns: int
    left_jpeg: bytes
    right_jpeg: bytes


def encode_frame_packet(frame_idx: int, timestamp_ns: int, left_jpeg: bytes, right_jpeg: bytes) -> bytes:
    if not left_jpeg or not right_jpeg:
        raise ValueError("left_jpeg and right_jpeg must be non-empty")
    header = HEADER_STRUCT.pack(
        MAGIC,
        VERSION,
        int(frame_idx),
        int(timestamp_ns),
        len(left_jpeg),
        len(right_jpeg),
    )
    return header + left_jpeg + right_jpeg


def decode_frame_packet(packet: bytes) -> FramePacket:
    if len(packet) < HEADER_STRUCT.size:
        raise ValueError("Packet too small to contain header")

    magic, version, frame_idx, timestamp_ns, left_size, right_size = HEADER_STRUCT.unpack_from(packet, 0)
    if magic != MAGIC:
        raise ValueError("Invalid packet magic")
    if version != VERSION:
        raise ValueError(f"Unsupported packet version: {version}")
    if left_size <= 0 or right_size <= 0:
        raise ValueError("Image payload sizes must be positive")

    payload = packet[HEADER_STRUCT.size :]
    expected_size = left_size + right_size
    if len(payload) != expected_size:
        raise ValueError(f"Payload size mismatch: expected {expected_size}, got {len(payload)}")

    left_jpeg = payload[:left_size]
    right_jpeg = payload[left_size:]
    return FramePacket(
        frame_idx=frame_idx,
        timestamp_ns=timestamp_ns,
        left_jpeg=left_jpeg,
        right_jpeg=right_jpeg,
    )

