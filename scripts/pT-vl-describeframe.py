## 按根据qwen3-vl-8b-instruct模型，生成对于单独每一幅画面帧的描述。
## 对每一帧的描述效果不赖
## 用于mimo-frame_prompt.txt和mimo-frame_gencot.py的生成

# 若总帧数过大，采取分阶段减小文本长度的采样，保证最终描述文本在45帧左右

import os
from pathlib import Path
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from datetime import datetime

# =========================
# 配置
# =========================
PROJECT_ROOT = "/root/zjf/bkbs-signcot"

DATA_ROOT = f"/root/autodl-tmp/z-csl-data"
MODEL_PATH = f"{DATA_ROOT}/models/Qwen/Qwen3-VL-8B-Instruct"
# FRAMES_ROOT = f"{DATA_ROOT}/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/train"
FRAMES_ROOT = f"{DATA_ROOT}/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/dev"

# 创建输出目录
OUTPUT_DIR = f"{PROJECT_ROOT}/prompts/phoenix-T/frame_descriptions_txt"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 创建log目录
LOG_DIR = f"{PROJECT_ROOT}/log"
os.makedirs(LOG_DIR, exist_ok=True)
# 日志文件路径
log_file = Path(LOG_DIR) / "describe-pTframe-log.jsonl"

MAX_NEW_TOKENS = 2048
DEVICE = "cuda"

# =========================
# 加载模型
# =========================
print("Loading model...")
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
print("Model loaded.")

# 支持的图像扩展名
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
    """对单张图像生成描述"""
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
# 主流程：遍历视频目录，逐帧生成描述并写入 .txt
# =========================
def main():    
    frame_root = Path(FRAMES_ROOT)
    if not frame_root.exists():
        raise FileNotFoundError(f"FRAMES_ROOT not found: {FRAMES_ROOT}")
    
    # 获取所有视频目录
    video_dirs = [d for d in sorted(frame_root.iterdir()) if d.is_dir()]
    total_videos = len(video_dirs)
    print(f"Found {total_videos} video directories to process.")
    
    for video_idx, video_dir in enumerate(video_dirs, start=1):
        print(f"\nProcessing video [{video_idx}/{total_videos}]: {video_dir.name}")
        frame_files = [f for f in video_dir.iterdir() if is_image_file(f)]
        if not frame_files:
            print(f"No images in {video_dir}, skipping.")
            continue
        
        # 按文件名排序（确保帧序正确）
        frame_files.sort(key=lambda x: x.name)

        output_lines = []
        frame_count = len(frame_files)
        successful_frames = 0
        error_frames = 0
        
        if frame_count > 200:
            continue  # 跳过过长视频
        
        for idx, frame_path in enumerate(frame_files, start=1):
            if frame_count < 110 and idx % 2 == 0: # 每2帧跳过1帧，(55, n)
                print(f"  [{idx}/{len(frame_files)}] Skipping {frame_path.name} (idx % 2 == 0)")
                continue  # 直接跳过，不处理这个帧
            if frame_count < 170 and frame_count >= 110 and idx % 3 != 1: # 每3帧跳过2帧，(57, 37)
                print(f"  [{idx}/{len(frame_files)}] Skipping {frame_path.name} (idx % 3 != 1)")
                continue  # 直接跳过，不处理这个帧
            if frame_count >= 170 and idx % 4 != 1: # 每4帧跳过3帧，(x, 43)
                print(f"  [{idx}/{len(frame_files)}] Skipping {frame_path.name} (idx % 4 != 1)")
                continue  # 直接跳过，不处理这个帧
            try:
                print(f"  [{idx}/{len(frame_files)}] Processing {frame_path.name} ...")
                desc = generate_description(str(frame_path))
                # 确保所有描述（包括置信度）都在一行内
                # 替换换行符为空格，确保一帧只占一行
                desc_one_line = desc.replace('\n', ' ').strip()
                # 去除多余空格
                while '  ' in desc_one_line:
                    desc_one_line = desc_one_line.replace('  ', ' ')
                output_lines.append(f"{idx}. {desc_one_line}")
                successful_frames += 1
                
            except Exception as e:
                error_msg = f"[ERROR: {str(e)}]"
                print(f"Error on {frame_path}: {e}")
                error_frames += 1
        
        # 写入 .txt 文件
        output_file = Path(OUTPUT_DIR) / f"{video_dir.name}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            # 在第一行添加总帧数信息
            f.write(f"(total {frame_count} frames in this video): \n")
            # 每行是一个完整的帧描述
            for line in output_lines:
                f.write(f"{line}\n")

        # 创建处理结果日志条目
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "video_name": video_dir.name,
            "video_index": video_idx,
            "total_videos": total_videos,
            "total_frames": frame_count,
            "processed_frames": len(output_lines),
            "successful_frames": successful_frames
        }
        
        # 写入日志文件
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        
        print(f"Saved descriptions to {output_file}")
        print(f"Total frames: {frame_count}, Processed frames: {len(output_lines)}")
        print(f"Log entry added to {log_file}")

    print("\nAll videos processed!")

if __name__ == "__main__":
    main()