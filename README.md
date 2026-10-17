# GTM: Rethinking CTC Alignment in CSLR

This is the official repository for the EMNLP 2026 main conference paper, “Beyond CTC Spikes: LLM-Driven Gloss-Temporal Supervision for Continuous Sign Language Recognition”.

In this work, we find that **dense supervision is more effective than sparse supervision**, **the idea of large model chain-of-thought can be transformed into intermediate supervision for deep learning models**, and **LLMs can be useful tools if prompt properly**. We introduce a novel GTM module to construct dense supervision, where the LLM serves as a reasoning operator rather than a sign language result generator. We further design a tailored curriculum learning strategy to guide the model from frame-level learning to sequence-level learning. Finally, without introducing additional trainable parameters, the resulting dense supervision achieves competitive performance against several state-of-the-art RGB-only methods and consistently improves the baseline across different visual backbones.

We provide all runnable code, prompt files, and the final annotation data. The annotation data must be used in accordance with the licenses of the original datasets (see [Data Preparation](#data-preparation)).

## Repo Directory

```bash
prompts
  ├─ Online_LLM_prompt.txt
scripts
  ├─ cslr-baseline.py（baseline）
  ├─ cslr-ce+ctc.py（ours）
  ├─ ctconlyspan.py（ctc anchor point）
  ├─ vl-describeframe.py（MLLM）
  ├─ frame-change.py（WDS, Window-based Difference Score）
  ├─ get-train-ann.py（Merge the above three types of info to form a txt file.）
  ├─ llm-reason.py（Online LLM）
  ├─ model-download.py
finalspan.txt
requirements-describe.txt
requirements-cslr.txt
```

------

## Prerequisites

Because the Python and PyTorch versions required by the MLLM and the CTC decoder are incompatible, we provide two separate `requirements.txt` files.

```bash
conda create -n gtm python=3.10
conda activate gtm
pip install -r requirements-describe.txt
```

```bash
conda create -n cslr python=3.8
conda activate cslr
pip install -r requirements-cslr.txt
```
- ctcdecode==0.4 [[parlance/ctcdecode](https://github.com/parlance/ctcdecode)]，for beam search decode.

------

## Data Preparation

- PHOENIX2014-T dataset: Download the RWTH-PHOENIX-Weather-2014-T Dataset [[download link](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)]
- CSL-Daily dataset: Request the CSL Dataset from this website [[download link](https://ustc-slr.github.io/datasets/2021_csl_daily/)]

Download datasets and extract them, no further data preprocessing needed.

### Data Usage Notice

We strictly comply with the data usage agreements established by the original dataset providers. Accordingly, csl_train_finalspan.txt is not publicly distributed. If you wish to use this file, please first obtain the necessary authorization from the original dataset team and then contact us.

------

## Steps

model-download -> GTM (ctc anchor point + MLLM + WDS -> get-train-ann -> Online LLM) -> finalspan.txt 

-> cslr-ce+ctc.py

------

## Cite

If you find our work useful or novel, please consider citing:

```bibtex
@inproceedings{
zhu2026beyond,
title={Beyond {CTC} Spikes: {LLM}-Driven Gloss-Temporal Supervision for Continuous Sign Language Recognition},
author={Junfeng Zhu and Cheng Jin and Yuchun Fang},
booktitle={The 2026 Conference on Empirical Methods in Natural Language Processing},
year={2026}
}
```