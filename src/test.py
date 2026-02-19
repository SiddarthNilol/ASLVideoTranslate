import os
from models.asl_inference import ASLInferenceSystem
from vjepa_encoder import VJEPA2Encoder

def example_single_video_inference(test_path):
    vjepa_encoder = VJEPA2Encoder(model_name="facebook/vjepa2-vitg-fpc64-256", device="cuda")

    # Initialize system
    system = ASLInferenceSystem(
        model_checkpoint_path='/home/dell/Desktop/ASLVideoTranslate/models/gloss_classifier_best.pt',
        vocab_path='/home/dell/Desktop/ASLVideoTranslate/models/vocab.json',
        vjepa_encoder=vjepa_encoder,
        device='cuda'
    )

    gloss, confidence = system.predict_from_video_file(test_path)

    print(f"Video: {test_path}")
    print(f"Predicted: {gloss}")
    print(f"Confidence: {confidence:.2%}")


def example_continuous_video_inference(test_path):
    """Example: Predict from continuous video with sliding window"""
    vjepa_encoder = VJEPA2Encoder(model_name="facebook/vjepa2-vitg-fpc64-256", device="cuda")
    system = ASLInferenceSystem(
        model_checkpoint_path='/home/dell/Desktop/ASLVideoTranslate/models/gloss_classifier_best.pt',
        vocab_path='/home/dell/Desktop/ASLVideoTranslate/models/vocab.json',
        vjepa_encoder=vjepa_encoder,
        device='cuda'
    )
    
    # Process continuous video
    predictions = system.predict_continuous_video(
        test_path,
        window_size=2.0,
        stride=1.0
    )
    
    print("\nDetected signs:")
    for pred in predictions:
        print(f"  {pred['timestamp']:.2f}s: {pred['gloss']} (conf: {pred['confidence']:.2f})")
    
    # Convert to sentence
    glosses = ' '.join([p['gloss'] for p in predictions])
    print(f"\nGloss sequence: {glosses}")


def example_end_to_end_translation(test_path):
    """
    Example: Complete end-to-end translation with T5
    """
    vjepa_encoder = VJEPA2Encoder(model_name="facebook/vjepa2-vitg-fpc64-256", device="cuda")
    # Initialize system with T5 translation enabled
    system = ASLInferenceSystem(
        model_checkpoint_path='/home/dell/Desktop/ASLVideoTranslate/models/gloss_classifier_best.pt',
        vocab_path='/home/dell/Desktop/ASLVideoTranslate/models/vocab.json',
        vjepa_encoder=vjepa_encoder,  # Your V-JEPA encoder
        device='cuda',
        use_t5_translation=True  # Enable T5
    )
    
    # Process video end-to-end
    result = system.predict_continuous_video_with_translation(
        test_path,
        window_size=2.0,
        stride=1.0,
        save_output=True,
        output_path='/home/dell/Desktop/ASLVideoTranslate/data/translation_output.json'
    )
    
    # Create captioned video
    _ = system.write_captioned_video(
        test_path,
        result['glosses'],
        result['english_translation'],
        output_path='/home/dell/Desktop/ASLVideoTranslate/data/captioned_video.mp4'
    )
    
    print("\n✓ Complete!")

if __name__ == "__main__":
    single_or_continuous = input("Test single video (s) or continuous video (c) or end-to-end translation (e)? [s/c/e]: ").strip().lower()
    if single_or_continuous not in ['s', 'c', 'e']:
        print("Invalid choice. Please enter 's' for single video, 'c' for continuous video, or 'e' for end-to-end translation.")
        exit(1)
    
    test_path = input("Enter path to test video: ")
    if not os.path.exists(test_path):
        print(f"File not found: {test_path}")
        exit(1)
    
    if single_or_continuous == 's':
        example_single_video_inference(test_path)
    elif single_or_continuous == 'e':
        example_end_to_end_translation(test_path)
    else:
        example_continuous_video_inference(test_path)