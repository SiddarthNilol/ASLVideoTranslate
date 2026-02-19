# File: streamlit_live_demo.py

import streamlit as st
import cv2
import torch
import numpy as np
from collections import deque
import time
from pathlib import Path
import json

import torchvision.transforms.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer

from models.asl_classifier import GlossClassifier
from vjepa_encoder import VJEPA2Encoder

# V-JEPA preprocessing constants
vjepa_num_frames = 16
vjepa_mean = (0.485, 0.456, 0.406)
vjepa_std = (0.229, 0.224, 0.225)


class LiveASLTranslator:
    """Real-time ASL translator for webcam feed"""
    
    def __init__(
        self,
        model_checkpoint_path,
        vocab_path,
        vjepa_encoder,
        device='cuda',
        window_size=2.0,
        use_t5=True
    ):
        self.device = device
        self.window_size = window_size
        self.fps = 30
        self.window_frames = int(window_size * self.fps)
        
        # Frame buffer for sliding window
        self.frame_buffer = deque(maxlen=self.window_frames)
        
        # Load vocabulary
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
        self.idx_to_gloss = vocab_data['idx_to_gloss']
        
        # Load gloss classifier
        checkpoint = torch.load(model_checkpoint_path, map_location=device)
        state_dict = checkpoint['model_state_dict']
        
        vjepa_dim = state_dict['attention_score.weight'].shape[1]
        num_classes = state_dict['classifier.8.weight'].shape[0]
        hidden_dim = state_dict['classifier.0.weight'].shape[0]
        
        self.model = GlossClassifier(
            vjepa_dim=vjepa_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes
        ).to(device)
        
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # V-JEPA encoder
        self.vjepa_encoder = vjepa_encoder
        
        # T5 for translation
        self.use_t5 = use_t5
        if use_t5:
            self.t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
            self.t5_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
            self.t5_model.eval()
        
        # State variables
        self.gloss_history = []
        self.current_gloss = "Waiting for sign..."
        self.current_confidence = 0.0
        self.english_translation = ""
        
        print("✓ Live ASL Translator initialized")
    
    def process_frame_buffer(self):
        """Process buffered frames and predict gloss"""
        if len(self.frame_buffer) < self.window_frames:
            return None, 0.0
        
        try:
            # Convert buffer to numpy array
            frames = np.array(list(self.frame_buffer))  # (T, H, W, 3)
            
            # Convert to tensor
            video_tensor = torch.from_numpy(frames).float().to(self.device)
            video_tensor = video_tensor.permute(0, 3, 1, 2)  # (T, 3, H, W)
            
            # Preprocess (crop, resize, normalize)
            H, W = video_tensor.shape[2], video_tensor.shape[3]
            min_dim = min(H, W)
            
            video = F.center_crop(video_tensor, [min_dim, min_dim])
            video = F.resize(video, [256, 256])
            
            T = video.shape[0]
            if T >= vjepa_num_frames:
                # Sample uniformly
                indices = torch.linspace(0, T - 1, vjepa_num_frames, device=self.device).long()
                video = video[indices]
                
                # Normalize
                vjepa_video = video / 255.0
                for c in range(3):
                    vjepa_video[:, c] = (vjepa_video[:, c] - vjepa_mean[c]) / vjepa_std[c]
                
                # Extract V-JEPA embedding
                vjepa_video = vjepa_video.to(self.vjepa_encoder.torch_dtype)
                
                with torch.no_grad():
                    embedding = self.vjepa_encoder.encode(vjepa_video)
                    
                    # Predict gloss
                    if embedding.ndim == 2:
                        embedding = embedding.unsqueeze(0)
                    
                    logits = self.model(embedding)
                    probs = torch.softmax(logits, dim=-1)
                    confidence, pred_idx = torch.max(probs, dim=-1)
                    
                    gloss = self.idx_to_gloss[str(pred_idx.item())]
                    
                    return gloss, confidence.item()
        
        except Exception as e:
            print(f"Error processing frame buffer: {e}")
            return None, 0.0
        
        return None, 0.0
    
    def translate_glosses(self, gloss_sequence):
        """Translate gloss sequence to English"""
        if not self.use_t5 or not gloss_sequence:
            return ""
        
        input_text = f"translate ASL gloss to English: {gloss_sequence}"
        
        inputs = self.t5_tokenizer(
            input_text,
            return_tensors='pt',
            max_length=128,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.t5_model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        
        english = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return english
    
    def update_gloss_history(self, gloss, confidence, threshold=0.6):
        """Update gloss history with deduplication"""
        # Only add high-confidence predictions
        if confidence < threshold:
            return
        
        # Avoid consecutive duplicates
        if self.gloss_history and self.gloss_history[-1] == gloss:
            return
        
        self.gloss_history.append(gloss)
        
        # Keep last 20 glosses
        if len(self.gloss_history) > 20:
            self.gloss_history.pop(0)
        
        # Update translation
        gloss_sequence = ' '.join(self.gloss_history[-10:])  # Last 10 for translation
        self.english_translation = self.translate_glosses(gloss_sequence)


def test_camera_connection(camera_source):
    """Test if camera/stream is accessible"""
    try:
        cap = cv2.VideoCapture(camera_source)
        ret, frame = cap.read()
        cap.release()
        return ret
    except:
        return False


def main():
    st.set_page_config(
        page_title="ASL Live Translator",
        page_icon="🤟",
        layout="wide"
    )
    
    # Title - CHANGED
    st.title("🤟 Live ASL Translation on GB10")
    st.markdown("Real-time American Sign Language translation powered by V-JEPA 2")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Camera source selection
    st.sidebar.subheader("📹 Camera Source")
    camera_type = st.sidebar.radio(
        "Select Camera Type",
        ["USB Webcam", "DroidCam (Phone)", "IP Camera"]
    )
    
    if camera_type == "USB Webcam":
        camera_source = st.sidebar.selectbox("USB Camera ID", [0, 1, 2], index=0)
    elif camera_type == "DroidCam (Phone)":
        phone_ip = st.sidebar.text_input(
            "Phone IP Address",
            value="10.21.83.65",
            help="Find this in DroidCam app on your phone"
        )
        phone_port = st.sidebar.text_input("Port", value="4747")
        camera_source = f"http://{phone_ip}:{phone_port}/video"
        
        # Test connection button
        if st.sidebar.button("🔍 Test Connection"):
            with st.spinner("Testing connection..."):
                if test_camera_connection(camera_source):
                    st.sidebar.success("✓ Connected to phone!")
                else:
                    st.sidebar.error("✗ Cannot connect. Check IP and make sure DroidCam app is running.")
    else:  # IP Camera
        camera_source = st.sidebar.text_input(
            "Camera URL",
            value="rtsp://192.168.1.100:8554/stream",
            help="Enter RTSP or HTTP stream URL"
        )
    
    st.sidebar.markdown("---")
    
    # Model configuration
    st.sidebar.subheader("🤖 Model Settings")
    
    model_path = st.sidebar.text_input(
        "Model Path",
        value="/home/dell/Desktop/ASLVideoTranslate/models/gloss_classifier_best.pt"
    )
    
    vocab_path = st.sidebar.text_input(
        "Vocabulary Path",
        value="/home/dell/Desktop/ASLVideoTranslate/models/vocab.json"
    )
    
    use_t5 = st.sidebar.checkbox("Enable T5 Translation", value=True)
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05
    )
    
    # Initialize session state
    if 'translator' not in st.session_state:
        st.session_state.translator = None
    
    if 'running' not in st.session_state:
        st.session_state.running = False
    
    if 'camera_source' not in st.session_state:
        st.session_state.camera_source = camera_source
    
    # Load model button
    if st.sidebar.button("🔄 Load Model"):
        with st.spinner("Loading model and V-JEPA encoder..."):
            try:
                # Load V-JEPA encoder
                # TODO: Implement your V-JEPA encoder loading
                # from your_vjepa_module import load_vjepa_encoder
                vjepa_encoder = VJEPA2Encoder(model_name="facebook/vjepa2-vitg-fpc64-256", device="cuda")
                
                
                # Uncomment when V-JEPA is ready:
                st.session_state.translator = LiveASLTranslator(
                    model_checkpoint_path=model_path,
                    vocab_path=vocab_path,
                    vjepa_encoder=vjepa_encoder,
                    device='cuda',
                    use_t5=use_t5
                )
                st.session_state.camera_source = camera_source
                st.success("✓ Model loaded successfully!")
                
            except Exception as e:
                st.error(f"Error loading model: {e}")
    
    # CHANGED: Single column layout (removed right sidebar)
    st.subheader("📹 Live Feed")

    # Control buttons
    button_col1, button_col2, button_col3 = st.columns(3)
    
    with button_col1:
        start_button = st.button("▶️ Start", use_container_width=True)
    
    with button_col2:
        stop_button = st.button("⏹️ Stop", use_container_width=True)
    
    with button_col3:
        clear_button = st.button("🗑️ Clear History", use_container_width=True)

    video_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # CHANGED: Translation info below video with bigger font
    st.markdown("---")
    english_placeholder = st.empty()
    history_placeholder = st.empty()
    
    # Handle buttons
    if start_button:
        st.session_state.running = True
        st.session_state.camera_source = camera_source
    
    if stop_button:
        st.session_state.running = False
    
    if clear_button and st.session_state.translator:
        st.session_state.translator.gloss_history = []
        st.session_state.translator.english_translation = ""
    
    # Main loop
    if st.session_state.running and st.session_state.translator:
        # Open camera/stream
        cap = cv2.VideoCapture(st.session_state.camera_source)
        
        if not cap.isOpened():
            status_placeholder.error(f"❌ Cannot open camera source: {st.session_state.camera_source}")
            st.session_state.running = False
        else:
            status_placeholder.success(f"✓ Connected to: {st.session_state.camera_source}")
            
            frame_count = 0
            prediction_interval = 15  # Predict every 15 frames (0.5 seconds)
            
            try:
                while st.session_state.running:
                    ret, frame = cap.read()
                    if not ret:
                        status_placeholder.warning("⚠️ Failed to grab frame")
                        time.sleep(0.1)
                        continue
                    
                    # Add frame to buffer
                    resized = cv2.resize(frame, (224, 224))
                    rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    st.session_state.translator.frame_buffer.append(rgb_frame)
                    
                    # Predict periodically
                    if (frame_count % prediction_interval == 0 and 
                        len(st.session_state.translator.frame_buffer) == 
                        st.session_state.translator.window_frames):
                        
                        gloss, confidence = st.session_state.translator.process_frame_buffer()
                        
                        if gloss and confidence > confidence_threshold:
                            st.session_state.translator.current_gloss = gloss
                            st.session_state.translator.current_confidence = confidence
                            st.session_state.translator.update_gloss_history(
                                gloss, confidence, confidence_threshold
                            )
                    
                    # CHANGED: Display clean frame (no overlays)
                    display_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(display_frame_rgb, channels="RGB", width=640) 
                    
                    # CHANGED: Update translation below video with bigger font
                    if st.session_state.translator.english_translation:
                        english_placeholder.markdown(
                            f"## 📝 English Translation\n### {st.session_state.translator.english_translation}",
                            unsafe_allow_html=True
                        )
                    else:
                        english_placeholder.markdown(
                            "## 📝 English Translation\n### *Waiting for signs...*"
                        )
                    
                    # CHANGED: Show gloss history with bigger font
                    if st.session_state.translator.gloss_history:
                        history_text = " → ".join(st.session_state.translator.gloss_history[-10:])
                        history_placeholder.markdown(
                            f"## 🔤 Sign History\n### {history_text}"
                        )
                    else:
                        history_placeholder.markdown(
                            "## 🔤 Sign History\n### *No signs detected yet*"
                        )
                    
                    frame_count += 1
                    
                    # Small delay to prevent overwhelming the UI
                    time.sleep(0.03)
            
            finally:
                cap.release()
                status_placeholder.info("⏸️ Stopped")
    
    elif st.session_state.running and not st.session_state.translator:
        st.warning("⚠️ Please load the model first!")
        st.session_state.running = False
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 System Info")
    
    if st.session_state.translator:
        st.sidebar.success(f"""
        ✓ Model loaded  
        ✓ Vocabulary: {len(st.session_state.translator.idx_to_gloss)} signs  
        ✓ T5 Translation: {'Enabled' if st.session_state.translator.use_t5 else 'Disabled'}
        """)
    else:
        st.sidebar.warning("⚠️ Model not loaded")
    
    # Instructions
    with st.expander("ℹ️ How to Use"):
        st.markdown("""
        ### Setup Instructions
        
        **For DroidCam (Phone as Webcam):**
        1. Install DroidCam app on your phone
        2. Connect phone and GB10 to same WiFi network
        3. Open DroidCam app and note the IP address
        4. Enter the IP in the sidebar
        5. Click "Test Connection" to verify
        
        **Usage:**
        1. Load the model using "Load Model" button
        2. Select your camera source
        3. Click "Start" to begin translation
        4. Sign in front of the camera
        5. Watch real-time translation appear below the video!
        
        **Tips:**
        - Keep signs within camera frame
        - Sign clearly with good lighting
        - Wait 0.5s between signs for best accuracy
        - Adjust confidence threshold if getting too many/few detections
        """)


if __name__ == "__main__":
    main()
