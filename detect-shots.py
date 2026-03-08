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
from PIL import Image
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

def build_embedding_model():
     model = models.resnet50(pretrained=True)
     model = nn.Sequential(*list(model.children())[:-1])  # Remove final classification layer
     model.eval()
     return model

def get_frame_embedding(frame, model, device="cpu"):
     frame_unit8 = (frame*255).astype(np.uint8) if frame.dtype == np.float32 else frame.astype(np.uint8)
     pil_image = Image.fromarray(frame_unit8)

     #normalize to imagenet stats and 224x224
     transform = Compose([Resize((224, 224)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
     frame_tensor = transform(pil_image).unsqueeze(0).to(device)

     with torch.no_grad():
         embedding = model(frame_tensor)
     return embedding.flatten()

def remove_dupliacte_shots(frames, shots, similarity_threshold=0.9):
    if len(shots) == 0:
        return shots
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Removing duplicate shots...")
    embedding_model = build_embedding_model().to(device)
    # extracting from middle frame from each of the shots
    embeddings = []
    for idx, (start,end) in enumerate(shots):
        mid_framed_idx = (start + end) // 2
        frame = frames[mid_framed_idx]
        embedding = get_frame_embedding(frame, embedding_model, device)
        embeddings.append(embedding)
        print(f"\r  Processed {idx+1}/{len(shots)} shots", end="")
    print()

     # Compute pairwise cosine similarity
    print("Computing similarity matrix...")
    n_shots = len(embeddings)
    similarity_matrix = np.zeros((n_shots, n_shots))
    
    for i in range(n_shots):
        for j in range(i, n_shots):
            # Cosine similarity = 1 - cosine distance
            similarity = 1 - cosine(embeddings[i], embeddings[j])
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity
    
    # Group shots by similarity and keep only the first from each group
    print("Grouping similar shots...")
    kept_shots = []
    used = set()
    
    for i in range(n_shots):
        if i in used:
            continue
        kept_shots.append(shots[i])
        used.add(i)
        # Mark all similar shots as used
        for j in range(i + 1, n_shots):
            if similarity_matrix[i, j] > similarity_threshold:
                used.add(j)
    
    print(f"Removed {len(shots) - len(kept_shots)} duplicate shots")
    return kept_shots

if __name__ == "__main__":
    frames = extract_frames(VIDEO_PATH)
    preds = run_transnetv2(frames)
    shots = predictions_to_shots(preds, 0.5)

    with open("out_shots.json", 'w') as fh:
                json.dump(shots, fh)
    print("Shot list written to out_shots.json")