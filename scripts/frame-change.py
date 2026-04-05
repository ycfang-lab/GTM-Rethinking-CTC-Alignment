## 基于 ResNet50 提取视频帧特征，计算变化分数，选出变化最大的帧

import os
os.environ["TORCH_HOME"] = "/root/autodl-tmp/z-csl-data/models/resnet"

from glob import glob
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models


# ============================================================
# Configuration
# ============================================================

FPS = 25
MIN_GAP = 4          # 相邻 5 帧中只能算 1 次

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Utilities
# ============================================================

def load_frames(frame_dir):
    frame_paths = sorted(
        glob(os.path.join(frame_dir, "*.png")) +
        glob(os.path.join(frame_dir, "*.jpg"))
    )
    if len(frame_paths) == 0:
        raise RuntimeError("No frames found in directory")
    return frame_paths


# ============================================================
# Feature extractor
# ============================================================

class ResNet50Feature(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1
        )
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        feat = self.backbone(x)
        return feat.flatten(1)


def extract_features(frame_paths, batch_size=16):
    model = ResNet50Feature().to(DEVICE).eval()

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    feats = []
    with torch.no_grad():
        for i in range(0, len(frame_paths), batch_size):
            imgs = []
            for p in frame_paths[i:i + batch_size]:
                img = Image.open(p).convert("RGB")
                imgs.append(transform(img))
            imgs = torch.stack(imgs).to(DEVICE)
            f = model(imgs)
            feats.append(f.cpu().numpy())

    return np.concatenate(feats, axis=0)


# ============================================================
# Change score
# ============================================================

def compute_change_score(features, w):
    """
    features: (T, D)
    w: dynamic window size
    return: change score, shape (T-1,)
    """
    v = features[1:] - features[:-1]
    Tm1 = len(v)
    change = np.zeros(Tm1, dtype=np.float32)

    if w * 2 >= Tm1:
        return change  # window 太大，直接返回全 0，避免越界

    for t in range(w, Tm1 - w):
        left = v[t - w:t]
        right = v[t:t + w]
        change[t] = np.linalg.norm(
            left.mean(axis=0) - right.mean(axis=0)
        )

    # 平滑
    change = np.convolve(change, np.ones(5) / 5, mode="same")
    return change


# ============================================================
# Select top change frames
# ============================================================

def select_top_change_frames(change, num_gloss, min_gap=MIN_GAP):
    """
    从 change 中选出变化最大的 (num_gloss + extra) 个帧号
    且任意两个结果至少相隔 min_gap 帧
    extra = min(num_gloss, 2)
    """
    extra = 3
    K = num_gloss + extra

    idx = np.argsort(change)[::-1]

    selected = []
    for i in idx:
        frame_id = i + 1  # change[i] 对应帧 i+1

        if all(abs(frame_id - s) >= min_gap for s in selected):
            selected.append(frame_id)

        if len(selected) >= K:
            break

    return sorted(selected)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    frame_dir = (
        "/root/autodl-tmp/z-csl-data/PHOENIX-2014-T-release-v3/"
        "PHOENIX-2014-T/features/fullFrame-210x260px/train/"
        "31October_2009_Saturday_tagesschau-7148"
    )

    glosses = ["MORGEN", "SECHS", "GRAD", "SIEBZEHN", "GRAD"]  # 示例 gloss 列表

    frames = load_frames(frame_dir)
    num_frames = len(frames)
    num_gloss = len(glosses)

    # 动态 window
    window = max(2, num_frames // (num_gloss + 5))

    print(f"Total frames: {num_frames}")
    print(f"Num glosses: {num_gloss}")
    print(f"Dynamic WINDOW: {window} frames (~{window / FPS:.2f}s)")

    features = extract_features(frames)
    change = compute_change_score(features, w=window)

    top_frames = select_top_change_frames(
        change,
        num_gloss=num_gloss
    )

    print("\nTop change frames (frame indices):")
    for f in top_frames:
        print(f)
