# ASL Video Translation on Dell Prox Max with GB10

Real-time American Sign Language (ASL) video translation system powered by V-JEPA 2 foundation model and edge computing on Dell Pro Max GB10.

## Overview

ASLVideoTranslate is a complete end-to-end system that:
- Encodes ASL videos using **V-JEPA 2** (Video Joint Embedding Predictive Architecture) for robust spatiotemporal feature extraction
- Classifies video embeddings to ASL glosses using a **lightweight attention-pooling classifier**
- Translates ASL gloss sequences to natural English using **Qwen2.5-3B-Instruct LLM**
- Provides **real-time translation** via Streamlit web interface with webcam/phone camera support
- Runs **100% on-device** on Dell Pro Max GB10 - no cloud dependencies

## Key Features

✅ **V-JEPA 2 Video Foundation Model**: State-of-the-art pretrained encoder for motion understanding  
✅ **Lightweight Attention Classifier**: Only ~1M parameters, trains in 1-2 hours  
✅ **LLM-Powered Translation**: Natural English generation with proper grammar  
✅ **Real-time Performance**: 10x faster than real-time processing  
✅ **Edge Computing**: Entire pipeline runs locally on GB10 (no internet required)  
✅ **Live Demo Interface**: Streamlit app with DroidCam phone camera support  
✅ **Privacy-First**: All video processing stays on-device  
✅ **Captioned Video Export**: Generate videos with burned-in subtitles  

## System Requirements

- **Hardware**: Dell Pro Max GB10 with 128GB unified memory
- **Software**: 
  - Python 3.9+
  - CUDA-capable GPU
  - FFmpeg (for video processing)
  - Optional: DroidCam for phone camera streaming

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ASLVideoTranslate.git
cd ASLVideoTranslate
```

### 2. Create Virtual Environment
```bash
python3 -m venv env
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install transformers
pip install opencv-python decord pillow
pip install streamlit
pip install scikit-learn pandas tqdm
```

### 4. Install FFmpeg (for video export)
```bash
sudo apt update
sudo apt install ffmpeg
```

### 5. Download Models

The system will automatically download:
- V-JEPA 2 model: `facebook/vjepa2-vitg-fpc64-256`
- Qwen2.5-3B-Instruct: `Qwen/Qwen2.5-3B-Instruct`

## Project Structure
```
ASLVideoTranslate/
├── data/
│   ├── WLASL_v0.3.json            # WLASL dataset 
│   ├── videos/                    # Raw ASL videos 
│   ├── selected_videos/           # Processed videos + embeddings
│   │   └── index.csv              # Dataset index
|
├── models/
│   ├── gloss_classifier_best.pt   # Trained classifier checkpoint
│   └── vocab.json                 # Gloss vocabulary mapping
├── src/ (or root)
├── models/
│       |── asl_classifier.py          # Lightweight gloss classifier
│       └── asl_inference.py           # Inference system with LLM 
│   ├── vjepa_encoder.py           # V-JEPA 2 encoder wrapper
│   ├── dataset.py                 # PyTorch dataset for embeddings
│   ├── train.py                   # Training script
│   ├── test.py                    # Testing utilities
│   └── streamlit_demo.py          # Live demo web interface
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Prepare Dataset (WLASL)

Download WLASL dataset:
```bash
# Download WLASL videos and metadata
# Available at: https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed

# Place videos in data/videos/
# Place WLASL_v0.3.json in data/
```

### 2. Process Videos and Extract V-JEPA Embeddings
```bash
python process_videos.py
```

This will:
- Load videos from `data/videos/`
- Extract V-JEPA 2 embeddings (shape: `[2048, 1408]`)
- Apply data augmentation (rotation, brightness, contrast)
- Save `.npy` embeddings to `data/selected_videos/`
- Create `index.csv` for training

### 3. Train the Classifier
```bash
python train.py --batch 32 --epochs 25 --lr 1e-3
```

**Training Configuration:**
- **Dataset**: ~7,300 videos, 300 unique glosses
- **Batch Size**: 16 (or 32 if memory allows)
- **Epochs**: 25-30
- **Learning Rate**: 1e-3 with cosine annealing
- **Expected Accuracy**: 95% (train), 90% (val)
- **Training Time**: 1-2 hours on GB10

**Arguments:**
- `--batch`: Batch size (default: 8)
- `--epochs`: Number of epochs (default: 20)
- `--lr`: Initial learning rate (default: 1e-3)

Model checkpoints saved to: `models/gloss_classifier_best.pt`

### 4. Test Inference

#### Single Video Inference
```bash
python test.py
# Choose 's' for single video
# Enter path to test video
```

#### Continuous Video with Translation
```bash
python test.py
# Choose 'e' for end-to-end
# Enter path to video (30-60 seconds)
```

Outputs:
- Detected glosses with timestamps
- English translation
- Captioned video (MP4 with burned-in subtitles)

### 5. Run Live Demo
```bash
streamlit run streamlit_demo.py
```

**Setup for DroidCam (Phone as Webcam):**
1. Install DroidCam app on phone
2. Connect phone and GB10 to same WiFi
3. Note phone IP from DroidCam app
4. Enter IP in Streamlit sidebar
5. Click "Test Connection"
6. Load model and start signing!

## Model Architecture

### Stage 1: V-JEPA 2 Encoder (Frozen)
```
Input: Video frames [T, 224, 224, 3]
  ↓
V-JEPA 2 (ViT-Giant, 1B params)
  ↓
Output: Embeddings [2048, 1408]
```

### Stage 2: Lightweight Gloss Classifier (~2M params)
```
Input: V-JEPA Embeddings [B, 2048, 1408]
  ↓
Attention Pooling:
  - Attention Score: Linear(1408 → 1)
  - Softmax over 2048 tokens
  - Weighted sum → [B, 1408]
  ↓
MLP Classifier:
  - Linear(1408 → 512) + LayerNorm + GELU + Dropout
  - Linear(512 → 256) + LayerNorm + GELU + Dropout
  - Linear(256 → 300) [num_classes]
  ↓
Output: Gloss logits [B, 300]
```

**Why Attention Pooling?**
- Learns which video tokens (spatial-temporal regions) are important
- Much smaller than transformer decoder (2M vs 50M params)
- Faster training and inference
- Less prone to overfitting with limited data

### Stage 3: LLM Translation (Qwen2.5-3B)
```
Input: Gloss sequence "HELLO MY NAME J-O-H-N"
  ↓
Qwen2.5-3B-Instruct (3.8B params, FP16)
  ↓
Output: "Hello, my name is John."
```


## Data Augmentation

**Applied during training:**
- **Rotation**: ±10° random rotation
- **Brightness**: ±15 random brightness adjustment
- **Contrast**: 0.85-1.15x random contrast
- **Horizontal Flip**: 50% chance (mirrors signer)


## Performance Metrics

### Classification Accuracy
- **Training**: 90-95%
- **Validation**: 85-90%
- **Model Size**: ~1M trainable parameters (classifier only)

### Inference Speed (GB10)
- **Single video**: ~50ms per 2-second clip
- **Continuous video**: 10x faster than real-time
- **LLM translation**: ~100-200ms per sentence
- **End-to-end**: ~5 seconds for 60-second video

### Memory Usage (GB10)
- V-JEPA 2: ~8GB
- Classifier: ~2GB
- Qwen2.5-3B: ~8GB (FP16)
- Total: ~18GB / 128GB available ✅

## File Formats

### V-JEPA Embeddings
```python
# Shape: (2048, 1408)
# Dtype: float32
# Format: .npz
embedding = np.load('video_vjepa.npz')['data']  # [2048, 1408]
```

### Index CSV
```csv
video_id,gloss,path_to_npy_file
hello_001,hello,/path/to/hello_001_vjepa.npy
goodbye_042,goodbye,/path/to/goodbye_042_vjepa.npy
```

### Vocabulary JSON
```json
{
  "gloss_to_idx": {"hello": 0, "goodbye": 1, ...},
  "idx_to_gloss": {"0": "hello", "1": "goodbye", ...},
  "num_classes": 300
}
```


## Contact

For questions, collaboration, or support:
- Open an issue on GitHub
- Email: sk12590@nyu.edu

---

**Built with ❤️ for the differently abled community**