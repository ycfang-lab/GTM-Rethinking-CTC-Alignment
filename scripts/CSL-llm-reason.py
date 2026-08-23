## Use a large-model API to integrate Qwen3-VL descriptions for each frame and generate corresponding JSON files
## Use framedes_prompt.txt

# Retries are triggered in the following cases:
# 1. JSON cannot be extracted
# 2. The number of glosses does not match the number of segments
# A maximum of 4 attempts are made in total (1 initial attempt + 3 retries), and each retry calls the API again to generate a completely new response

# Cases that still fail after multiple retries and manual extraction are manually processed using the web-based LLM.

## The API key should be placed in the environment variables

# Each successfully processed training sample takes about 30-40s; CSL requires about 28 days, and pT about 12 days

import os
import json
import re
from datetime import datetime
from openai import OpenAI


output_dir = f"/cotdata/CSL"
os.makedirs(output_dir, exist_ok=True)

frame_desc_root = f"/prompts/CSL/frame_descriptions_txt"
csl_annotation_path = f"/prompts/CSL/csl_train_annotation.txt"

log_dir = f"/log"
os.makedirs(log_dir, exist_ok=True)
log_file = f"{log_dir}/CSLgcot-log.jsonl"

# Read the prompt file
with open(f"/prompts/CSL-frame_prompt.txt", "r", encoding="utf-8") as f:
    base_prompt = f.read().strip()

# Read and parse the csl_train_annotation.txt file
def parse_csl_annotation(file_path):
    entries = []

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    blocks = re.split(r'(?m)^={20,}\s*$', text)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        entry = {}

        name_match = re.search(
            r'(?m)^name:\s*(.*?)\s*$',
            block
        )
        if name_match:
            entry["name"] = name_match.group(1).strip()

        gloss_match = re.search(
            r'(?m)^gloss_sequence:\s*(.*?)\s*$',
            block
        )
        if gloss_match:
            entry["gloss_sequence"] = gloss_match.group(1).strip()

        video_structure_match = re.search(
            r'(?ms)^video_structure:\s*\n?(.*)$',
            block
        )
        if video_structure_match:
            entry["video_structure"] = video_structure_match.group(1).strip()

        if all(
            key in entry
            for key in ["name", "gloss_sequence", "video_structure"]
        ):
            entries.append(entry)

    return entries

def load_each_frame_description(entry, frame_desc_root):
    """
    Load the frame-by-frame description text corresponding to entry["name"]
    from the frame_descriptions_txt directory
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
    Check whether the corresponding {name}.json file for entry["name"]
    already exists in output_dir
    """
    name = entry.get("name")
    if not name:
        return True  # If even the name is missing, treat it as existing and filter it out
    
    json_path = os.path.join(output_dir, f"{name}.json")
    return os.path.exists(json_path)

def extract_json_from_response(response_text):
    """
    Extract the JSON part from the response text
    Handle the following cases:
    1. Contains ```json ... ``` markers
    2. Directly returns a JSON object
    """
    # Try to match the ```json ... ``` format
    json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # If there is no code block marker, use the entire response directly
        json_str = response_text.strip()
    
    # Try to parse JSON
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        # print(f"Original response: {response_text[:500]}...")
        return None

def count_glosses_in_sequence(gloss_sequence):
    """
    Count the number of glosses in gloss_sequence
    Format: "1]glossA 2]glossB ..."
    """    
    # Split by spaces and filter out empty strings
    parts = [part for part in gloss_sequence.split() if part]
    
    # Count valid glosses (items starting with a number followed by ])
    gloss_count = 0
    for part in parts:
        if re.match(r'^\d+\]', part):
            gloss_count += 1
    
    return gloss_count

def count_segments_in_json(json_data):
    """
    Count the number of items in reasoning.segments in the JSON data
    """
    try:
        if "reasoning" in json_data and "segments" in json_data["reasoning"]:
            return len(json_data["reasoning"]["segments"])
        else:
            return 0
    except (KeyError, TypeError):
        return 0

# Load parsed entries
entries = parse_csl_annotation(csl_annotation_path)

filtered_entries = []
for entry in entries:
    # Check whether the output JSON file already exists
    if check_output_json_exists(entry, output_dir):
        continue  # Skip entries that have already been processed
    
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
    """Write logs in JSONL format"""
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_data, ensure_ascii=False) + "\n")

for i, entry in enumerate(entries):
    # Write the processing-start log
    process_start_log = {
        "timestamp": datetime.now().isoformat(),
        "level": "INFO", 
        "message": f"Processing entry {i+1}/{len(entries)}: {entry['name']}"
    }
    write_jsonl_log(process_start_log, log_file)
    print(f"Processing entry {i+1}/{len(entries)}: {entry['name']}")

    # Build the complete prompt
    full_prompt = f"{base_prompt}\n"
    # full_prompt += "Input: \n\n"
    # full_prompt += f"name: {entry['name']}\n\n"
    full_prompt += f"gloss_sequence(positioning_point): {entry['gloss_sequence']}\n"
    full_prompt += f"video_structure: {entry['video_structure']}\n"
    full_prompt += f"each_frame_description:\n{entry['each_frame_description']}"
    
    # Retry counter
    retry_count = 0
    max_retries = 4  # Maximum of 4 attempts in total (1 initial attempt + 3 retries)
    success = False
    while retry_count < max_retries and not success:
        try:        
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
        
            # Record the successful API response log
            api_success_log = {
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "type": "api_success",
                "entry_name": entry['name'],
                "retry_count": retry_count+1,
            }
            write_jsonl_log(api_success_log, log_file)
        
            print(f"Attempt {retry_count+1}: API response succeeded")
            # print(response_content)
        
            # Extract JSON data
            json_data = extract_json_from_response(response_content)
            
            # Retry if JSON cannot be extracted
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
                    # Retry limit reached; record the error
                    json_error_log = {
                        "timestamp": datetime.now().isoformat(),
                        "level": "ERROR",
                        "type": "json_parse_error",
                        "entry_name": entry['name'],
                        "retry_count": retry_count,
                    }
                    write_jsonl_log(json_error_log, log_file)
                    error_msg = f"Failed to parse the response for entry {entry['name']} as JSON"
                    print(error_msg)
                    break
            
            # Check consistency between the number of glosses and segments
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
                    print(f"Gloss count ({gloss_count}) does not match segment count ({segment_count}); retrying")
                    continue
                else:
                    # Retry limit reached; record the error
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
                    error_msg = f"Entry {entry['name']} has mismatched gloss count ({gloss_count}) and segment count ({segment_count})"
                    print(error_msg)
                    break
        
            # Consistency check passed; save the JSON file
            name = entry['name']
            output_file = f"{output_dir}/{name}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)
            
            # Record the successful-save log
            save_success_log = {
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "type": "save_success",
                "entry_name": entry['name'],
            }
            write_jsonl_log(save_success_log, log_file)
            print(f"Successfully saved: {output_file}")
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
            error_msg = f"Error processing entry {entry['name']}: {str(e)}"
            print(error_msg)
            print("\n" + "="*30 + "\n")
            break

# Record the processing-complete log
end_log = {
    "timestamp": datetime.now().isoformat(),
    "level": "INFO",
    "type": "process_complete",
    "total_entries": len(entries),
}
write_jsonl_log(end_log, log_file)

# Record processing completion
completion_message = f"Processing completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
print(completion_message)