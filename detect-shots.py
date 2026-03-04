import sys
import argparse
import json
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from scipy.spatial.distance import cosine
import ffmpeg

TRANSNET_WEIGHTS = "weights/transnetv2-pytorch-weights.pth"
VIDEO_PATH = "test/warhammer.mp4"

from transnetv2_pytorch import TransNetV2

def extract_frames(video_path):
    print(f"Extracting frames from {video_path}...")
    video_stream, _ = (
        ffmpeg.input(video_path)
        .output("pipe:", format="rawvideo", pix_fmt="rgb24", s="48x27")
        .run(capture_stdout=True, capture_stderr=True)
    )
    frames = np.frombuffer(video_stream, np.uint8).reshape([-1, 27, 48, 3])
    print(f"  Total frames: {len(frames)}")
    return frames

def run_transnetv2(frames):
    print("Loading TransNetV2...")
    model = TransNetV2()
    state_dict = torch.load(TRANSNET_WEIGHTS, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # Pad frames
    pad_start = np.tile(frames[0:1], (25, 1, 1, 1))
    pad_end_n = 25 + (50 - len(frames) % 50) % 50
    pad_end = np.tile(frames[-1:], (pad_end_n, 1, 1, 1))
    padded = np.concatenate([pad_start, frames, pad_end], axis=0)

    predictions = []
    print(f"Running inference ({len(frames)} frames)...")
    with torch.no_grad():
        for i in range(0, len(padded) - 100 + 1, 50):
            window = padded[i:i+100]
            inp = torch.from_numpy(window).unsqueeze(0)
            single, _ = model(inp)
            single = torch.sigmoid(single).squeeze().numpy()
            predictions.append(single[25:75])
            print(f"\r  Processed {min((i//50 + 1)*50, len(frames))}/{len(frames)} frames", end="")
    print()

    return np.concatenate(predictions)[:len(frames)]

def predictions_to_shots(preds, threshold):
    is_boundary = (preds > threshold).astype(np.uint8)
    shots, start = [], 0
    for i in range(1, len(is_boundary)):
        if is_boundary[i] == 1 and is_boundary[i-1] == 0:
            shots.append((start, i - 1))
            start = i
    shots.append((start, len(preds) - 1))
    return shots


if __name__ == "__main__":
    frames = extract_frames(VIDEO_PATH)
    preds = run_transnetv2(frames)
    shots = predictions_to_shots(preds, 0.5)

    with open("out_shots.json", 'w') as fh:
                json.dump(shots, fh)
    print("Shot list written to out_shots.json")