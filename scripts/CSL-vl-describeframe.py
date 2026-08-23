## Generate descriptions for each individual video frame using the Qwen3-VL-8B-Instruct model.

# If the total number of frames is too large, use staged sampling to reduce text length, ensuring the final description text contains about 40 frames
# On a single RTX 4090, describing every frame would take about 50 days; using the sampling strategy in this code takes about 30 days

import os
import csv
from pathlib import Path
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from datetime import datetime

# =========================
# Configuration
# =========================
MODEL_PATH = f"/models/Qwen/Qwen3-VL-8B-Instruct"
FRAMES_ROOT = f"/CSL-Daily/frames_512x512"

name_CSV = f"/CSL-Daily/sentence_label/train_name.csv" # Manually extracted train-only video names from split_1.csv

# Create output directory
OUTPUT_DIR = f"/prompts/CSL/frame_descriptions_txt"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create log directory
LOG_DIR = f"/log"
os.makedirs(LOG_DIR, exist_ok=True)
# Log file path
log_file = Path(LOG_DIR) / "describe-CSLframe-log.jsonl"

MAX_NEW_TOKENS = 2048
DEVICE = "cuda"

# =========================
# Load model
# =========================
print("Loading model...")
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
print("Model loaded.")

# Supported image extensions
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def is_image_file(filepath):
    return Path(filepath).suffix.lower() in IMG_EXTENSIONS

PROMPT = (
    "Describe this image." 
    "Pay special attention to the hands of the characters in the picture (gestures, hand shapes, relative positions of the hands, palm orientation, etc.)," 
    "and you can infer the speed of hand movement based on the degree of motion blur in the picture(the more blurred the motion, the faster it is)." 
    "Also pay attention to the non-hand parts of the characters (facial expressions(especially special eyebrow and mouth shapes), body postures)." 
    "Only describe body posture if it is NOT upright and facing forward/audience (e.g., if there is a tendency or tilt, etc.)."
    "Only describe shoulders if they are NOT relaxed (e.g., if they are tensed, raised, slouched, or asymmetrical)."
    "Do not describe the 'character attributes', 'portraits', and 'background' in the picture." 
    "Do not fabricate, completely based on the content of the image. If it is really unclear, you can write 'unclear', or 'estimate the hand based on the position of the arm'." 
    "You can also write sentences like 'there are no hands in the picture' or 'the left hand is not in the picture'." 
    "And after the final description, add a confidence value (0-1). The higher the value, the greater your confidence in this description."
)

def generate_description(image_path):
    """Generate a description for a single image"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": PROMPT}
            ],
        }
    ]
    
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        return_tensors="pt",
    ).to(DEVICE)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False
        )
    
    generated_text = processor.batch_decode(
        generated_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    return generated_text.strip()

# =========================
# Main process: traverse video directories, generate descriptions frame by frame, and write them to .txt files
# =========================
def main():    
    frame_root = Path(FRAMES_ROOT)
    if not frame_root.exists():
        raise FileNotFoundError(f"FRAMES_ROOT not found: {FRAMES_ROOT}")
    
    # Read training set names from the CSV file
    print(f"Reading training names from {name_CSV}...")
    train_names = set()
    try:
        with open(name_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="|")
            for row in reader:
                name = row.get("name", "").strip()
                if name:
                    train_names.add(name)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        print("Will process all videos as fallback.")
        train_names = None  # If reading fails, process all videos
    
    print(f"Found {len(train_names) if train_names else 'all'} video names in CSV.")
    
    # Get all video directories and filter them
    video_dirs = []
    for d in sorted(frame_root.iterdir()):
        if d.is_dir():
            # If train_names is unavailable (reading failed), or the folder name is in train_names
            if train_names is None or d.name in train_names:
                video_dirs.append(d)
            # else:
            #     print(f"Skipping {d.name} (not in training set)")
    
    total_videos = len(video_dirs)
    if total_videos == 0:
        print("No matching video directories found!")
        # if train_names:
        #     print("First 10 names in train_names:")
        #     for name in list(train_names)[:10]:
        #         print(f"  {name}")
        return
    
    print(f"Found {total_videos} video directories to process (from training set).")
    
    for video_idx, video_dir in enumerate(video_dirs, start=1):
        print(f"\nProcessing video [{video_idx}/{total_videos}]: {video_dir.name}")
        frame_files = [f for f in video_dir.iterdir() if is_image_file(f)]
        if not frame_files:
            print(f"No images in {video_dir}, skipping.")
            continue
        
        # Sort by filename to ensure correct frame order
        frame_files.sort(key=lambda x: x.name)

        output_lines = []
        frame_count = len(frame_files)
        successful_frames = 0
        error_frames = 0
        
        for idx, frame_path in enumerate(frame_files, start=1): # Redundant processing; if time permits, describe every frame by commenting out this entire section
            if frame_count < 160 and idx % 3 != 1: # Skip 2 out of every 3 frames, (53, n)
                print(f"  [{idx}/{len(frame_files)}] Skipping {frame_path.name} (idx % 3 != 1)")
                continue  # Skip this frame directly without processing
            if frame_count >= 160 and idx % 4 != 1: # Skip 3 out of every 4 frames, (x, 40)
                print(f"  [{idx}/{len(frame_files)}] Skipping {frame_path.name} (idx % 4 != 1)")
                continue  # Skip this frame directly without processing
            try:
                print(f"  [{idx}/{len(frame_files)}] Processing {frame_path.name} ...")
                desc = generate_description(str(frame_path))
                # Ensure all descriptions (including confidence) remain on a single line
                # Replace newline characters with spaces so each frame occupies only one line
                desc_one_line = desc.replace('\n', ' ').strip()
                # Remove redundant spaces
                while '  ' in desc_one_line:
                    desc_one_line = desc_one_line.replace('  ', ' ')
                output_lines.append(f"{idx}. {desc_one_line}")
                successful_frames += 1
                
            except Exception as e:
                error_msg = f"[ERROR: {str(e)}]"
                print(f"Error on {frame_path}: {e}")
                error_frames += 1
        
        # Write to .txt file
        output_file = Path(OUTPUT_DIR) / f"{video_dir.name}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            # Add total frame count information on the first line
            f.write(f"(total {frame_count} frames in this video): \n")
            # Each line contains a complete frame description
            for line in output_lines:
                f.write(f"{line}\n")

        # Create processing result log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "video_name": video_dir.name,
            "video_index": video_idx,
            "total_videos": total_videos,
            "total_frames": frame_count,
            "processed_frames": len(output_lines),
            "successful_frames": successful_frames
        }
        
        # Write to log file
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        
        print(f"Saved descriptions to {output_file}")
        print(f"Total frames: {frame_count}, Processed frames: {len(output_lines)}")
        print(f"Log entry added to {log_file}")

    print("\nAll videos processed!")

if __name__ == "__main__":
    main()