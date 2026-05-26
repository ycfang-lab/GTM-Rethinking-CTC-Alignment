## 用大模型api，整合qwen3-vl对每一帧的描述，生成对应的gcot表示并保存为json文件
## 用mimo-frame_prompt.txt

# 当遇到以下情况时会触发重试：
# 1.无法提取JSON
# 2.gloss数量与segment数量不一致
# 总共最多尝试4次（第1次 + 3次重试），每次重试都会重新调用api生成全新的响应内容

## 要把apikey放入environ

# 每成功处理一条训练数据约30-40s，CSL需要约28天，pT约12天

import os
import json
import re
from datetime import datetime
from openai import OpenAI

PROJECT_ROOT = "/root/zjf/bkbs-signcot"

output_dir = f"{PROJECT_ROOT}/cotdata/CSL"
os.makedirs(output_dir, exist_ok=True)

frame_desc_root = f"{PROJECT_ROOT}/prompts/CSL/frame_descriptions_txt"
csl_annotation_path = f"{PROJECT_ROOT}/prompts/CSL/csl_train_annotation.txt"

log_dir = f"{PROJECT_ROOT}/log"
os.makedirs(log_dir, exist_ok=True)
log_file = f"{log_dir}/mimo-CSLgcot-log.jsonl"

# 读取prompt文件
with open(f"{PROJECT_ROOT}/prompts/CSL-mimo-frame_prompt.txt", "r", encoding="utf-8") as f:
    base_prompt = f.read().strip()

# 读取并解析csl_train_annotation.txt文件
def parse_csl_annotation(file_path):
    pass
        
    # 确保所有字段都存在
    if all(key in entry for key in ["name", "gloss_sequence", "video_structure"]):
        entries.append(entry)
    return entries

def load_each_frame_description(entry, frame_desc_root):
    """
    从 frame_descriptions_txt 目录中加载与 entry["name"] 对应的逐帧描述文本
    """
    name = entry.get("name")
    if not name:
        return None

    txt_path = os.path.join(frame_desc_root, f"{name}.txt")
    if not os.path.exists(txt_path):
        return None

    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read().strip()
    
def check_output_json_exists(entry, output_dir):
    """
    检查 output_dir 中是否已存在与 entry["name"] 对应的 {name}.json 文件
    """
    name = entry.get("name")
    if not name:
        return True  # 如果连名字都没有，就当它存在（过滤掉）
    
    json_path = os.path.join(output_dir, f"{name}.json")
    return os.path.exists(json_path)

def extract_json_from_response(response_text):
    """
    从响应文本中提取JSON部分
    处理以下情况：
    1. 包含```json ... ```标记
    2. 直接返回JSON对象
    """
    # 尝试匹配 ```json ... ``` 格式
    json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # 如果没有代码块标记，直接使用整个响应
        json_str = response_text.strip()
    
    # 尝试解析JSON
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        # print(f"原始响应: {response_text[:500]}...")
        return None

def count_glosses_in_sequence(gloss_sequence):
    """
    统计gloss_sequence中的词数量
    形式: "1]glossA 2]glossB ..."
    """    
    # 按空格分割并过滤空字符串
    parts = [part for part in gloss_sequence.split() if part]
    
    # 统计有效的gloss（以数字]开头的内容）
    gloss_count = 0
    for part in parts:
        if re.match(r'^\d+\]', part):
            gloss_count += 1
    
    return gloss_count

def count_segments_in_json(json_data):
    """
    统计JSON数据中reasoning.segments的数量
    """
    try:
        if "reasoning" in json_data and "segments" in json_data["reasoning"]:
            return len(json_data["reasoning"]["segments"])
        else:
            return 0
    except (KeyError, TypeError):
        return 0

# 加载解析的条目
entries = parse_csl_annotation(csl_annotation_path)

filtered_entries = []
for entry in entries:
    # 检查是否已存在输出 JSON 文件
    if check_output_json_exists(entry, output_dir):
        continue  # 跳过已处理过的条目
    
    each_frame_description = load_each_frame_description(entry, frame_desc_root)
    if each_frame_description is not None:
        entry["each_frame_description"] = each_frame_description

    if all(key in entry for key in ["name", "gloss_sequence", "video_structure", "each_frame_description"]):
        filtered_entries.append(entry)

entries = filtered_entries

client = OpenAI(
    api_key=os.environ.get("API_KEY"),
    base_url="BASE_URL"
)

def write_jsonl_log(log_data, log_file):
    """写入JSONL格式的日志"""
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_data, ensure_ascii=False) + "\n")

for i, entry in enumerate(entries):
    # 写入处理开始日志
    process_start_log = {
        "timestamp": datetime.now().isoformat(),
        "level": "INFO", 
        "message": f"开始处理第 {i+1}/{len(entries)} 个条目: {entry['name']}"
    }
    write_jsonl_log(process_start_log, log_file)
    print(f"处理第 {i+1}/{len(entries)} 个条目: {entry['name']}")

    # 构建完整的prompt
    full_prompt = f"{base_prompt}\n"
    # full_prompt += "Input: \n\n"
    # full_prompt += f"name: {entry['name']}\n\n"
    full_prompt += f"gloss_sequence(positioning_point): {entry['gloss_sequence']}\n"
    full_prompt += f"video_structure: {entry['video_structure']}\n"
    full_prompt += f"each_frame_description:\n{entry['each_frame_description']}"
    
    # 重试计数器
    retry_count = 0
    max_retries = 4  # 总共最多尝试4次（第1次 + 3次重试）
    success = False
    while retry_count < max_retries and not success:
        try:
            # # 如果是重试，记录重试信息
            # if retry_count > 0:
            #     retry_log = {
            #         "timestamp": datetime.now().isoformat(),
            #         "level": "WARNING",
            #         "type": "api_retry",
            #         "entry_name": entry['name'],
            #         "retry_count": retry_count,
            #     }
            #     write_jsonl_log(retry_log, log_file)
            #     print(f"第{retry_count}次重新API调用: {entry['name']}")
        
            completion = client.chat.completions.create(
                model="deepseek-V3",
                messages=[
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ],
                max_completion_tokens=3072,
                temperature=0.3,
                top_p=0.95,
                stream=False,
                stop=None,
                frequency_penalty=0,
                presence_penalty=0,
            )
        
            response_content = completion.choices[0].message.content
        
            # 记录API响应成功日志
            api_success_log = {
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "type": "api_success",
                "entry_name": entry['name'],
                "retry_count": retry_count+1,
            }
            write_jsonl_log(api_success_log, log_file)
        
            print(f"第{retry_count+1}次尝试响应，api成功")
            # print(response_content)
        
            # 提取JSON数据
            json_data = extract_json_from_response(response_content)
            
            # 如果无法提取JSON，进行重试
            if json_data is None:
                if retry_count < max_retries - 1:
                    retry_count += 1
                    json_retry_log = {
                        "timestamp": datetime.now().isoformat(),
                        "level": "WARNING",
                        "type": "json_retry",
                        "entry_name": entry['name'],
                        "retry_count": retry_count,
                    }
                    write_jsonl_log(json_retry_log, log_file)
                    continue
                else:
                    # 重试次数用完，记录错误
                    json_error_log = {
                        "timestamp": datetime.now().isoformat(),
                        "level": "ERROR",
                        "type": "json_parse_error",
                        "entry_name": entry['name'],
                        "retry_count": retry_count,
                    }
                    write_jsonl_log(json_error_log, log_file)
                    error_msg = f"无法解析条目 {entry['name']} 的响应为JSON"
                    print(error_msg)
                    break
            
            # 检查gloss数量与segment数量的一致性
            gloss_count = count_glosses_in_sequence(entry['gloss_sequence'])
            segment_count = count_segments_in_json(json_data)
            
            if gloss_count != segment_count:
                if retry_count < max_retries - 1:
                    retry_count += 1
                    consistency_retry_log = {
                        "timestamp": datetime.now().isoformat(),
                        "level": "WARNING",
                        "type": "consistency_retry",
                        "entry_name": entry['name'],
                        "gloss_count": gloss_count,
                        "segment_count": segment_count,
                        "retry_count": retry_count,
                    }
                    write_jsonl_log(consistency_retry_log, log_file)
                    print(f"gloss数量({gloss_count})与segment数量({segment_count})不一致，进行重试")
                    continue
                else:
                    # 重试次数用完，记录错误
                    consistency_error_log = {
                        "timestamp": datetime.now().isoformat(),
                        "level": "ERROR",
                        "type": "consistency_error",
                        "entry_name": entry['name'],
                        "error_type": "gloss_segment_mismatch",
                        "gloss_count": gloss_count,
                        "segment_count": segment_count,
                        "retry_count": retry_count,
                    }
                    write_jsonl_log(consistency_error_log, log_file)
                    error_msg = f"条目 {entry['name']} 的gloss数量({gloss_count})与segment数量({segment_count})不一致"
                    print(error_msg)
                    break
        
            # 一致性检查通过，保存JSON文件
            name = entry['name']
            output_file = f"{output_dir}/{name}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)
            
            # 记录成功日志
            save_success_log = {
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "type": "save_success",
                "entry_name": entry['name'],
            }
            write_jsonl_log(save_success_log, log_file)
            print(f"成功保存: {output_file}")
            print("\n" + "="*30 + "\n")
            success = True

        except Exception as e:
            error_log = {
                "timestamp": datetime.now().isoformat(),
                "level": "ERROR",
                "type": "error",
                "entry_name": entry['name'],
                "error_type": type(e).__name__,
                "error_message": str(e),
            }
            write_jsonl_log(error_log, log_file)
            error_msg = f"处理条目 {entry['name']} 时出错: {str(e)}"
            print(error_msg)
            print("\n" + "="*30 + "\n")
            break

# 记录处理完成日志
end_log = {
    "timestamp": datetime.now().isoformat(),
    "level": "INFO",
    "type": "process_complete",
    "total_entries": len(entries),
}
write_jsonl_log(end_log, log_file)

# 记录处理完成
completion_message = f"处理完成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
print(completion_message)