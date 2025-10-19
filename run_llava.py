import os
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch

# Import necessary components
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria


CHECKPOINT_PATH = "lmms-lab/llava-onevision-qwen2-7b-ov"


def get_model_name_from_path(path):
    """Extract model name from checkpoint path."""
    return os.path.basename(path)

def initialize_models():
    """Initialize LLaVA model"""
    # Disable unnecessary PyTorch initialization
    disable_torch_init()

    # Load the pre-trained model
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        CHECKPOINT_PATH,
        model_name=CHECKPOINT_PATH,
        model_base=None,
        load_8bit=False,
        load_4bit=False
    )
    
    # Set model to evaluation mode and move to appropriate device
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    return tokenizer, model, image_processor

def process_image(image, tokenizer, model, image_processor, temperature=0.2):
    """Process a single image with LLaVA model to extract timestamp"""
    # Convert string path to PIL Image if needed
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    
    # Extract timestamp region from the bottom right corner
    img_width, img_height = image.size
    left_ratio, top_ratio = 0.75, 0.90
    right_ratio, bottom_ratio = 1.00, 1.00
    
    # Calculate crop coordinates
    left = int(img_width * left_ratio)
    top = int(img_height * top_ratio)
    right = int(img_width * right_ratio)
    bottom = int(img_height * bottom_ratio)
    
    # Crop timestamp region
    timestamp_img = image.crop((left, top, right, bottom))
    
    # Process timestamp region with LLaVA
    timestamp_prompt = "Please extract the text from image and output in JSON format. One output example is like this: {\"time\": 2025-01-03 10:30:00, \"coord\": [37.112342, 116.112342]}. Please follow the format strictly. If the coordinates are not available, please output None."
    
    # Get timestamp using LLaVA
    timestamp_img = timestamp_img.copy()  # Create a copy to avoid meta tensor issues
    timestamp = process_single_image(timestamp_img, tokenizer, model, image_processor, timestamp_prompt, temperature)
    
    return timestamp

def process_single_image(image, tokenizer, model, image_processor, prompt, temperature=0.2):
    """Helper function to process a single image with LLaVA"""
    # Process image and ensure it's on CUDA
    image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"].half()
    image_tensor = image_tensor.cuda()
    
    # Determine conversation mode based on model name
    model_name = get_model_name_from_path(CHECKPOINT_PATH)
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "qwen_1_5"
    
    # Setup conversation and handle image tokens
    conv = conv_templates[conv_mode].copy()
    if model.config.mm_use_im_start_end:
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt
    else:
        inp = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()
    
    input_ids = tokenizer_image_token(prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    
    # Setup stopping criteria
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
    
    # Generate response
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=temperature,
            top_p=1.0,
            max_new_tokens=512,
            use_cache=True,
            stopping_criteria=[stopping_criteria]
        )
        
        # Move output_ids to CPU before decoding
        outputs = tokenizer.batch_decode(output_ids.cpu(), skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
    
    return outputs

def process_directory(input_dir, output_dir, temperature=0.2):
    """Process all images in input_dir and save results to output_dir maintaining directory structure"""
    # Initialize model
    print("Initializing LLaVA model...")
    tokenizer, model, image_processor = initialize_models()
    
    # Convert to Path objects for easier handling
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Get all image files recursively
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [
        f for f in input_dir.rglob('*') 
        if f.suffix.lower() in image_extensions
    ]

    image_files = sorted(image_files)
    
    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            # Load and process image
            image = Image.open(image_path).convert('RGB')
            
            # Get relative path to maintain directory structure
            rel_path = image_path.relative_to(input_dir)
            
            # Create output directory structure
            output_path = output_dir / rel_path.parent
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Process image using LLaVA
            timestamp = process_image(
                image,
                tokenizer,
                model,
                image_processor,
                temperature=temperature
            )
            
            # Create result dictionary
            result = {
                'image_path': str(rel_path),
                'timestamp': timestamp
            }
            
            # Save JSON with same name as image but .json extension
            json_path = output_path / f"{rel_path.stem}_text.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")

if __name__ == "__main__":
    input_dir = "../test_datasets/simple"
    output_dir = "../test_datasets/simple_result"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all images
    process_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        temperature=0.2
    )
