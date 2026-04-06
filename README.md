# Repo Directory
```bash
prompts
  ├─ CSL/per_frame_description/
  ├─ phoenix-T/per_frame_description/
  ├─ Online_LLM_prompt.txt
scripts
  ├─ cslr-baseline.py（baseline）
  ├─ csle-ce+ctc.py（ours）
  ├─ ctconlyspan.py（ctc anchor point）
  ├─ vl-describeframe.py（MLLM）
  ├─ frame-change.py（WDS）
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
- PHOENIX2014-T dataset: Download the RWTH-PHOENIX-Weather 2014 Dataset [[download link](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)]
- CSL dataset: Request the CSL Dataset from this website [[download link](https://ustc-slr.github.io/openresources/cslr-dataset-2015/index.html)]
Download datasets and extract them, no further data preprocessing needed.
------
# Steps
GTM(ctc anchor point + MLLM + WDS + Online LLM) -> finalspan.txt 
-> csle-ce+ctc.py