import os
import torch
from threading import Thread
from PIL import Image
import gradio as gr
from transformers.generation.streamers import TextIteratorStreamer
import clip
from torch.nn.functional import cosine_similarity
from transformers import AutoImageProcessor, DetaForObjectDetection
import numpy as np
from PIL import ImageDraw
from PIL import ImageFont

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria

# Disable unnecessary PyTorch initialization
disable_torch_init()

# Load the pre-trained model
CHECKPOINT_PATH = "./checkpoints/llava-onevision-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-si_stage_am9_egocity"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    CHECKPOINT_PATH, model_name=CHECKPOINT_PATH, model_base=None, load_8bit=False, load_4bit=False
)

# DETA model initialization
deta_processor = AutoImageProcessor.from_pretrained("jozhang97/deta-swin-large")
deta_model = DetaForObjectDetection.from_pretrained("jozhang97/deta-swin-large")
deta_model = deta_model.to("cuda" if torch.cuda.is_available() else "cpu")

# CLIP model for deduplication
clip_model, clip_preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")

def get_model_name_from_path(path):
    # This is a placeholder implementation. You might want to implement a more robust model name extraction logic based on your file naming convention or by reading the model file.
    return os.path.basename(path)

def get_image_features(image, clip_model, clip_preprocess, device):
    image_input = clip_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
    return image_features

def process_image_with_deta(image, detection_threshold=0.3, similarity_threshold=0.9):
    """Extract person crops from image using DETA."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Prepare inputs and get model outputs
    inputs = deta_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = deta_model(**inputs)

    # Post-process results
    target_sizes = torch.tensor([image.size[::-1]])
    results = deta_processor.post_process_object_detection(outputs, threshold=detection_threshold, target_sizes=target_sizes)[0]

    width, height = image.size
    crops_with_boxes = []  # Now store both crop and its original box coordinates

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if deta_model.config.id2label[label.item()] == "person":
            box = [round(i, 2) for i in box.tolist()]
            center_x = (box[0] + box[2]) / 2
            center_y = 0.7 * box[1] + 0.3 * box[3]
            
            # Calculate expanded bbox
            box_width = box[2] - box[0]
            box_height = box[3] - box[1]
            expanded_box = [
                max(0, center_x - box_width),
                max(0, center_y - box_height),
                min(width, center_x + box_width),
                min(height, center_y + box_height)
            ]

            # Skip if box is too small
            if (expanded_box[2] - expanded_box[0]) * (expanded_box[3] - expanded_box[1]) < 0.002 * width * height:
                continue
            
            cropped_img = image.crop(expanded_box)
            crops_with_boxes.append((cropped_img, score.item(), expanded_box))

    # Deduplicate crops using CLIP
    saved_features = {}
    filtered_crops = []
    
    for idx, (cropped_img, score, box) in enumerate(crops_with_boxes):
        features = get_image_features(cropped_img, clip_model, clip_preprocess, device)
        
        is_duplicate = False
        for existing_features in saved_features.values():
            similarity = cosine_similarity(features, existing_features)
            if similarity > similarity_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            filtered_crops.append((cropped_img, box))
            saved_features[idx] = features

    # sort from left to right
    filtered_crops = sorted(filtered_crops, key=lambda x: x[1][0])  # Sort by x-coordinate of bounding box
    
    return filtered_crops

def model_inference(image, prompt, temperature=0.2, top_p=1.0, max_tokens=1024):
    """Processes the image and generates a response based on the prompt."""
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"].half().cuda()
    
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
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    
    # Generate response
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria]
        )
        
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
    
    return outputs

def model_inference_pipeline(image, temperature=0.2, detection_threshold=0.5, similarity_threshold=0.95):
    """Process image through DETA and LLaVA pipeline."""
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    
    # Get person crops with their original boxes
    crops_with_boxes = process_image_with_deta(image, detection_threshold, similarity_threshold)
    
    # Fixed prompt
    prompt = "请描述这张图片的中心的人物"
    
    # Create a copy of the original image for visualization
    vis_image = image.copy()
    draw = ImageDraw.Draw(vis_image)
    
    # Load font
    try:
        font = ImageFont.truetype("Arial Bold.ttf", 20)
    except:
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()
    
    def get_contrasting_color(img, x, y):
        # Sample a 5x5 region around the center point
        sample_size = 5
        half_size = sample_size // 2
        
        x, y = int(x), int(y)
        r_total, g_total, b_total = 0, 0, 0
        sample_count = 0
        
        for dx in range(-half_size, half_size + 1):
            for dy in range(-half_size, half_size + 1):
                sample_x = min(max(x + dx, 0), img.width - 1)
                sample_y = min(max(y + dy, 0), img.height - 1)
                
                pixel = img.getpixel((sample_x, sample_y))
                r_total += pixel[0]
                g_total += pixel[1]
                b_total += pixel[2]
                sample_count += 1
        
        r_avg = r_total // sample_count
        g_avg = g_total // sample_count
        b_avg = b_total // sample_count
        
        return (255 - r_avg, 255 - g_avg, 255 - b_avg)
    
    # Process each crop through LLaVA
    responses = []
    valid_person_count = 0
    valid_centers = []
    
    for idx, (crop, box) in enumerate(crops_with_boxes):
        response = model_inference(crop, prompt, temperature, 1.0, 1024)
        
        # Skip if the person is driving or if no person is detected
        if "驾驶" in response or "无人物" in response:
            continue
            
        valid_person_count += 1
        response_short = response.replace("这张图片的中心人物是", "")
        responses.append(f"{valid_person_count}号{response_short}")
        
        # Calculate center point in original image coordinates
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        valid_centers.append((center_x, center_y))
    
    # Draw numbers after collecting all valid detections
    for i, (center_x, center_y) in enumerate(valid_centers, 1):
        number_text = str(i)
        text_bbox = draw.textbbox((0, 0), number_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Calculate text position
        text_x = center_x - text_width/2
        text_y = center_y - text_height/2
        
        # Get complementary color based on background
        text_color = get_contrasting_color(vis_image, center_x, center_y)
        
        # Draw text with outline for better visibility
        outline_color = (0, 0, 0)  # black outline
        for offset_x, offset_y in [(-1,-1), (-1,1), (1,-1), (1,1)]:
            draw.text((text_x + offset_x, text_y + offset_y), 
                     number_text, 
                     fill=outline_color, 
                     font=font)
        
        # Draw the main text in complementary color
        draw.text((text_x, text_y), 
                 number_text, 
                 fill=text_color, 
                 font=font)
    
    if not responses:
        return "No valid people detected in the image.", vis_image
    
    return "图中总共检测到" + str(valid_person_count) + "个人。从左到右依次是：\n" + "\n".join(responses), vis_image

# Update Gradio interface
with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.Markdown("# EgoCity Demo with Pedestrian Detection (Xiamengwei Zhang)")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Input Image")
            
            with gr.Row():
                temperature_input = gr.Slider(0, 1, value=0.2, label="Temperature")
                detection_threshold_input = gr.Slider(0, 1, value=0.2, label="NMS Threshold")
                similarity_threshold_input = gr.Slider(0, 1, value=0.95, label="Sim. Cutoff")
            
            submit_btn = gr.Button("Submit")
        
        with gr.Column():
            output = gr.Textbox(label="Response")
            output_image = gr.Image(label="Visualization")
    
    # Set up example inputs
    gr.Examples(
        examples=[
            ["./data/images/0000.jpg"],
            ["./data/images/0032.jpg"],
            ["./data/images/0063.jpg"],
            ["./data/images/0009.jpg"],
            ["./data/images/0025.jpg"],
            ["./data/images/0032.jpg"],
            ["./data/images/0041.jpg"],
            ["./data/images/0045.jpg"],
            ["./data/images/0061.jpg"],
            ["./data/images/0080.jpg"],
            ["./data/images/0106.jpg"],
            ["./data/images/0140.jpg"],
            ["./data/images/0185.jpg"],
            ["./data/images/502.jpg"],
            ["./data/images/503.jpg"],
            ["./data/images/523.jpg"],
            ["./data/images/528.jpg"],
            ["./data/images/535.jpg"],
            ["./data/images/543.jpg"],
        ],
        inputs=[image_input]
    )
    
    # Update click event
    submit_btn.click(
        fn=model_inference_pipeline,
        inputs=[image_input, temperature_input, detection_threshold_input, similarity_threshold_input],
        outputs=[output, output_image]
    )

if __name__ == "__main__":
    demo.launch(share=True)
