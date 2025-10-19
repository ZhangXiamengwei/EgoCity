from PIL import Image
import easyocr
import numpy as np
import json
from tqdm import tqdm
from pathlib import Path
from openai import AzureOpenAI
import os
import time

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Process images to extract timestamps and coordinates')
    parser.add_argument('dir_id', type=str, default='1',
                       help='Directory containing input images and output files')
    parser.add_argument('accelerate', type=str, default='1',
                       help='Accelerate')
    return parser.parse_args()

args = parse_args()

dir_id = args.dir_id
accelerate = args.accelerate

input_dir = f"../test_0313/spring/春节后数据抽帧/{dir_id}"
output_dir = f"../process_data/{dir_id}"

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

# Filter out images that already have corresponding JSON files
filtered_image_files = []
for image_path in image_files:
    # Calculate the expected JSON output path
    rel_path = image_path.relative_to(input_dir)
    json_path = output_dir / rel_path.parent / f"{rel_path.stem}_text.json"
    
    # Only include images whose JSON doesn't exist yet
    if not json_path.exists():
        filtered_image_files.append(image_path)

print(f"{dir_id}: Found {len(filtered_image_files)} images with missing JSON files.")

# split the filtered_image_files into 10 parts, and use accelerate_id to determine which part to process
accelerate_id = int(accelerate)
filtered_image_files = filtered_image_files[accelerate_id::10]

print(f"{dir_id}: Processing {len(filtered_image_files)} images.")

# wait for one minute
time.sleep(60)

system_prompt = """Task:
    You are an information transcription assistant responsible for processing OCR results, 
    which contain recognized bounding boxes and extracted content. Your objective is to 
    accurately extract and record timestamps and geographic coordinates from the provided OCR results.

    Instructions:
    1. **Error Correction:** The initial OCR results may contain recognition errors, 
       such as misplaced decimal points, extra spaces, or misformatted values. Use 
       common sense and standard formatting conventions to correct these errors.

    2. **Output Format:** Convert the corrected information into the following structured JSON format:
    
       {
         "date": "YYYY-MM-DD",
         "time": "HH:MM:SS",
         "coord": {
           "N": latitude,
           "E": longitude
         }
       }
    
    3. **Example:**
       **Input OCR:** 
       ['2025-01-24/14:36:57', 'N:34', '128651 E: 116.507462']
       
       **Corrected Output JSON:**
       {
         "date": "2025-01-24",
         "time": "14:36:57",
         "coord": {
           "N": 34.128651,
           "E": 116.507462
         }
       }
    
    Your task is to process a given OCR result and return only the correctly formatted JSON output.
    """

def get_timestamp(timestamp_coords):    
    # Initialize Azure OpenAI client
    endpoint = os.getenv("ENDPOINT_URL", "https://egoasr.openai.azure.com/")
    deployment = os.getenv("DEPLOYMENT_NAME", "egodataset")
    subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "5b496f4100d14ce58186e1bfe91db644")

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=subscription_key,
        api_version="2024-05-01-preview"
    )

    # Prepare chat prompt
    chat_prompt = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": f"Extract timestamp and coordinates from this OCR text: {timestamp_coords}"
                }
            ]
        }
    ]

    # Generate completion
    completion = client.chat.completions.create(
        model=deployment,
        messages=chat_prompt,
        max_tokens=800,
        temperature=0.2,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False
    )
    # Parse response
    try:
        raw_content = completion.choices[0].message.content

        # Strip markdown code block markers
        if raw_content.startswith("```json") and raw_content.endswith("```"):
            raw_content = raw_content[7:-3].strip()  # Remove ```json and closing ```

        # Parse JSON
        timestamp_json = json.loads(raw_content)
        return timestamp_json
    except:
        return {"time": None, "coord": None}



for image_path in tqdm(filtered_image_files, desc="Processing missing images"):
    try:
        # Load and process image
        img = Image.open(image_path).convert('RGB')

        # 获取图像的宽度和高度
        img_width, img_height = img.size

        # 定义裁剪区域的比例（例如红框区域）
        left_ratio = 0.75  # 左侧比例
        top_ratio = 0.90  # 上侧比例
        right_ratio = 1.00  # 右侧比例
        bottom_ratio = 1.00  # 下侧比例

        # 根据比例计算裁剪区域的坐标
        left = int(img_width * left_ratio)
        top = int(img_height * top_ratio)
        right = int(img_width * right_ratio)
        bottom = int(img_height * bottom_ratio)

        # 裁剪图像
        cropped_img = img.crop((left, top, right, bottom))

        # Convert PIL Image to numpy array for easyocr
        cropped_array = np.array(cropped_img)

        # 使用pytesseract进行OCR
        reader = easyocr.Reader(['en'])
        text = reader.readtext(cropped_array)

        # Extract timestamp and coordinates from OCR results
        timestamp_coords = []
        for detection in text:
            # Skip first line which contains noise
            if detection[1].startswith('HS'):
                continue
            timestamp_coords.append(detection[1])

        # load GPT-4o-mini to get the timestamp
        timestamp = get_timestamp(timestamp_coords)

        # Get relative path to maintain directory structure
        rel_path = image_path.relative_to(input_dir)
        
        # Create output directory structure
        output_path = output_dir / rel_path.parent
        output_path.mkdir(parents=True, exist_ok=True)

        # Save JSON with same name as image but .json extension
        json_path = output_path / f"{rel_path.stem}_text.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(timestamp, f, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
