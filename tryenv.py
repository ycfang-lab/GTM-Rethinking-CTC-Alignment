import torch
import numpy
from PIL import Image
import transformers
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# from modelscope import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import modelscope

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("numpy:", numpy.__version__)
print("transformers:", transformers.__version__)
print(modelscope.__version__)
