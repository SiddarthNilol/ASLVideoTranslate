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

from vjepa_encoder import VJEPA2Encoder

VJEPA_NUM_FRAMES = 16
VJEPA_MEAN = (0.485, 0.456, 0.406)
VJEPA_STD = (0.229, 0.224, 0.225)

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

    step = 1
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
    

def apply_gaussian_blur(frames: np.ndarray) -> np.ndarray:
    """Apply Gaussian blur to each frame."""
    blurred_frames = []
    for frame in frames:
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        blurred_frames.append(blurred)
    return np.stack(blurred_frames, axis=0)


def convert_to_black_and_white(frames: np.ndarray) -> np.ndarray:
    """Convert frames to black and white."""
    bw_frames = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        bw = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        bw_frames.append(bw)
    return np.stack(bw_frames, axis=0)


def process_video(data_dir="/home/dell/Desktop/ASLVideoTranslate/data", video_dir="/home/dell/Desktop/ASLVideoTranslate/data/videos", output_dir="/home/dell/Desktop/ASLVideoTranslate/data/selected_videos"):
    os.makedirs(output_dir, exist_ok=True)

    print("Initializing VJEPA encoder...")
    vjepa_encoder = VJEPA2Encoder(model_name=VJEPA_MODEL_NAME, torch_dtype=torch.bfloat16, device=DEVICE)
    
    #Read top gloss list from a .txt file
    with open('/home/dell/Desktop/ASLVideoTranslate/data/top_glosses.txt', 'r') as f:
        top_gloss_list = [line.strip() for line in f.readlines()]
    
    json_index_file = json.load(open(os.path.join(data_dir, "WLASL_v0.3.json"), "r"))

    for entry in tqdm(json_index_file):
        gloss = entry['gloss']
        if gloss not in top_gloss_list:
            continue  # Skip glosses that are not in the top list
                     
        for instance in entry['instances']:
            video_id = instance['video_id']
            video_path = os.path.join(video_dir, f"{video_id}.mp4")
            if not os.path.exists(video_path):
                continue
            
            frames = load_video_opencv(video_path, target_fps=30)

            gaussian_frames = apply_gaussian_blur(frames)
            bw_frames = convert_to_black_and_white(gaussian_frames)

            all_frames_list = [frames, gaussian_frames, bw_frames]
            idx_list = ["orig", "gaussian", "bw"]
            for frames, idx in zip(all_frames_list, idx_list):
                video_tensor = torch.from_numpy(frames).float().to(DEVICE)  # (T, H, W, 3)
                video_tensor = video_tensor.permute(0, 3, 1, 2) # (T, 3, H, W)

                H_orig = video_tensor.shape[2]
                W_orig = video_tensor.shape[3]
                min_dim = min(H_orig, W_orig)

                video = F.center_crop(video_tensor, [min_dim, min_dim])
                video = F.resize(video, [256, 256])

                T = video.shape[0]
                if T < VJEPA_NUM_FRAMES:
                    continue  # Skip videos that are too short

                indices = torch.linspace(0, T - 1, VJEPA_NUM_FRAMES, device=DEVICE).long()
                video = video[indices]

                vjepa_video = video / 255.0
                for c in range(3):
                    vjepa_video[:, c] = (vjepa_video[:, c] - VJEPA_MEAN[c]) / VJEPA_STD[c]

                vjepa_video = vjepa_video.to(vjepa_encoder.torch_dtype)

                vjepa_arr = vjepa_encoder.encode(vjepa_video)
                vjepa_arr = vjepa_arr.cpu().numpy()
                vjepa_ouput_path = os.path.join(output_dir, f"{gloss}_{video_id}_{idx}_vjepa.npz")
                np.savez_compressed(vjepa_ouput_path, data=vjepa_arr)


def create_index_file(processed_dir="/home/dell/Desktop/ASLVideoTranslate/data/selected_videos", index_file="/home/dell/Desktop/ASLVideoTranslate/data/WLASL_v0.3.json"):
    selected_files = glob(os.path.join(processed_dir, "*_vjepa.npz"))
    index_list = []
    for file_path in selected_files:
        filename = os.path.basename(file_path)
        parts = filename.split("_")
        if len(parts) < 4:
            print(f"Warning: Unexpected filename format {filename}, skipping.")
            continue
        gloss = parts[0]

        index_list.append({
            "video_id": filename,
            "gloss": gloss,
            "path_to_npy_file": file_path
        })
    index_df = pd.DataFrame(index_list)
    index_df.to_csv(os.path.join(processed_dir, "index.csv"), index=False)


if __name__ == "__main__":
    process_video()
    create_index_file()