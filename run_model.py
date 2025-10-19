import os
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
import clip
from transformers import AutoImageProcessor, DetaForObjectDetection

# Import necessary components from gradio_demo
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria

# Import the model inference function from gradio_demo
from gradio_demo import model_inference_pipeline, process_image_with_deta, model_inference

def initialize_models():
    """Initialize all required models following gradio_demo setup"""
    # Disable unnecessary PyTorch initialization
    disable_torch_init()

    # Load the pre-trained model
    CHECKPOINT_PATH = "./checkpoints/llava-onevision-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-si_stage_am9_egocity"
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        CHECKPOINT_PATH, 
        model_name=CHECKPOINT_PATH, 
        model_base=None, 
        load_8bit=False, 
        load_4bit=False
    )

    # DETA model initialization
    deta_processor = AutoImageProcessor.from_pretrained("jozhang97/deta-swin-large")
    deta_model = DetaForObjectDetection.from_pretrained("jozhang97/deta-swin-large")
    deta_model = deta_model.to("cuda" if torch.cuda.is_available() else "cpu")

    # CLIP model for deduplication
    clip_model, clip_preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")

    return tokenizer, model, image_processor, deta_processor, deta_model, clip_model, clip_preprocess

def process_directory(input_dir, output_dir, temperature=0.2, detection_threshold=0.5, similarity_threshold=0.95):
    """
    Process all images in input_dir and save results to output_dir maintaining directory structure
    """
    # Initialize all models first
    print("Initializing models...")
    tokenizer, model, image_processor, deta_processor, deta_model, clip_model, clip_preprocess = initialize_models()
    
    # Convert to Path objects for easier handling
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Get all image files recursively
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [
        f for f in input_dir.rglob('*') 
        if f.suffix.lower() in image_extensions
    ]
    
    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Get relative path to maintain directory structure
            rel_path = image_path.relative_to(input_dir)
            
            # Create output directory structure
            output_path = output_dir / rel_path.parent
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Process image using the pipeline
            description, annotated_image = model_inference_pipeline(
                image,
                temperature=temperature,
                detection_threshold=detection_threshold,
                similarity_threshold=similarity_threshold
            )
            
            # Create result dictionary
            result = {
                'description': description,
                'image_path': str(rel_path)
            }
            
            # Save JSON with same name as image but .json extension
            json_path = output_path / f"{rel_path.stem}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
                
            # Save annotated image
            annotated_image_path = output_path / f"{rel_path.stem}_annotated{rel_path.suffix}"
            annotated_image.save(annotated_image_path)
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")

if __name__ == "__main__":
    # input_dir = "../test_0313/simple"
    # output_dir = "../test_0313/simple_results"
    input_dir = f"../test_0313/spring/春节后数据抽帧/zhongguancun/0226"
    output_dir = f"../process_data/zhongguancun/0226"
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all images
    process_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        temperature=0.2,
        detection_threshold=0.2,
        similarity_threshold=0.95
    )
