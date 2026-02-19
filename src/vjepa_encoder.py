"""VJEPA2 encoder class.

Expose a simple function `forward_vjepa_video(model, video, ...)` that returns patch-wise features for the input video.
The function accepts either a model name string.
"""

from __future__ import annotations
import numpy as np
import torch
from transformers import AutoVideoProcessor, AutoModel
from typing import Optional


class VJEPA2Encoder():
    def __init__(self, model_name: str, torch_dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None):
        self.device = device or torch.device("cpu")
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.transform = AutoVideoProcessor.from_pretrained(model_name)
        self.torch_dtype = torch_dtype

    def encode(self, video_tensor) -> torch.Tensor:
        with torch.inference_mode():
            x_hf = self.transform(video_tensor, return_tensors="pt")["pixel_values_videos"].to(self.device)
            
            # Extract the patch-wise features from the last layer
            out_patch_features_hf = self.model.get_vision_features(x_hf)

        return out_patch_features_hf