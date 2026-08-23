# Repo Directory
```bash
prompts
  ├─ Online_LLM_prompt.txt
scripts
  ├─ cslr-baseline.py（baseline）
  ├─ cslr-ce+ctc.py（ours）
  ├─ ctconlyspan.py（ctc anchor point）
  ├─ vl-describeframe.py（MLLM）
  ├─ frame-change.py（Window-based Difference Score）
  ├─ get-train-ann.py（Merge the above three types of info to form a txt file.）
  ├─ llm-reason.py（Online LLM）
  ├─ model-download.py
finalspan.txt
requirements.txt
```
------
# Prerequisites
```bash
conda create -n gtm python=3.10
conda activate gtm
pip install -r requirements.txt
```
- ctcdecode==0.4 [[parlance/ctcdecode](https://github.com/parlance/ctcdecode)]，for beam search decode.
------
# Data Preparation
- PHOENIX2014-T dataset: Download the RWTH-PHOENIX-Weather-2014-T Dataset [[download link](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)]
- CSL-Daily dataset: Request the CSL Dataset from this website [[download link](https://ustc-slr.github.io/datasets/2021_csl_daily/)]

Download datasets and extract them, no further data preprocessing needed.
------
# Steps
GTM (ctc anchor point + MLLM + WDS -> Online LLM) -> finalspan.txt 
-> csle-ce+ctc.py