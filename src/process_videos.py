import os
from glob import glob
import json
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms.functional as F
from tqdm import tqdm

from src.vjepa_encoder import VJEPA2Encoder

vjepa_num_frames = 16
vjepa_mean = (0.485, 0.456, 0.406)
vjepa_std = (0.229, 0.224, 0.225)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VJEPA_MODEL_NAME = "facebook/vjepa2-vitg-fpc64-256"

def load_video_opencv(path: str, target_fps: int) -> np.ndarray:
    """Load video using OpenCV.
    
    Returns:
        frames: (T, H, W, 3) uint8 numpy array
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps is None or orig_fps <= 0:
        orig_fps = target_fps

    step = max(1, int(round(orig_fps / target_fps)))
    frames = []
    frame_idx = 0
    target_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            # Convert BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        frame_idx += 1

    cap.release()
    if frames:
        return np.stack(frames, axis=0)
    else:
        raise RuntimeError(f"No frames extracted from {path}")

def process_video(video_dir="/home/dell/Desktop/ASLVideoTranslate/data/videos", output_dir="/home/dell/Desktop/ASLVideoTranslate/data/processed_videos"):

    vjepa_encoder = VJEPA2Encoder(model_name=VJEPA_MODEL_NAME, torch_dtype=torch.bfloat16, device=DEVICE)

    video_path_list = glob(os.path.join(video_dir, "*.mp4"))
    print(f"Found {len(video_path_list)} videos to process.")
    
    for video_path in tqdm(video_path_list):
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        frames = load_video_opencv(video_path, target_fps=30)
        video_tensor = torch.from_numpy(frames).float().to(DEVICE)  # (T, H, W, 3)
        video_tensor = video_tensor.permute(0, 3, 1, 2) # (T, 3, H, W)

        H_orig = video_tensor.shape[2]
        W_orig = video_tensor.shape[3]
        min_dim = min(H_orig, W_orig)

        video = F.center_crop(video_tensor, [min_dim, min_dim])
        video = F.resize(video, [256, 256])

        T = video.shape[0]
        indices = torch.linspace(0, T - 1, vjepa_num_frames, device=DEVICE).long()
        video = video[indices]

        vjepa_video = video / 255.0
        for c in range(3):
            vjepa_video[:, c] = (vjepa_video[:, c] - vjepa_mean[c]) / vjepa_std[c]

        vjepa_video = vjepa_video.to(vjepa_encoder.torch_dtype)

        vjepa_arr = vjepa_encoder.encode(vjepa_video)
        vjepa_ouput_path = os.path.join(output_dir, f"{base_name}_vjepa.npz")
        np.savez_compressed(vjepa_ouput_path, vjepa_arr)

def create_index_file(processed_dir="/home/dell/Desktop/ASLVideoTranslate/data/processed_videos", index_file="/home/dell/Desktop/ASLVideoTranslate/data/WLASL_v0.3.json"):
    npz_files = glob(os.path.join(processed_dir, "*_vjepa.npz"))

    index_list = []
    json_content = json.load(open(index_file, "r"))
    
    for entry in json_content:
        gloss = entry['gloss']
        instances = entry['instances']

        for inst in instances:
            video_id = inst['video_id']
            npz_path = os.path.join(processed_dir, f"{video_id}_vjepa.npz")
            if os.path.exists(npz_path):
                index_list.append({
                    "video_id": video_id,
                    "gloss": gloss,
                    "vjepa_path": npz_path
                })
            else:
                print(f"Warning: {npz_path} not found, skipping.")
    
    index_df = pd.DataFrame(index_list)
    index_df.to_csv(os.path.join(processed_dir, "index.csv"), index=False)

if __name__ == "__main__":
    process_video()
    create_index_file()