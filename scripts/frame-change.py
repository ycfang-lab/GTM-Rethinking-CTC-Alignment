## Extract video frame features using ResNet18, compute change scores, and select frames with the largest changes

import os
os.environ["TORCH_HOME"] = "/models/resnet"

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
MIN_GAP = 4          # Only one frame can be selected among every 5 adjacent frames

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

class ResNet18Feature(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        feat = self.backbone(x)
        return feat.flatten(1)


def extract_features(frame_paths, batch_size=16):
    model = ResNet18Feature().to(DEVICE).eval()

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
        return change  # If the window is too large, return all zeros directly to avoid out-of-bounds access

    for t in range(w, Tm1 - w):
        left = v[t - w:t]
        right = v[t:t + w]
        change[t] = np.linalg.norm(
            left.mean(axis=0) - right.mean(axis=0)
        )

    # Smoothing
    change = np.convolve(change, np.ones(5) / 5, mode="same")
    return change


# ============================================================
# Select top change frames
# ============================================================

def select_top_change_frames(change, num_gloss, min_gap=MIN_GAP):
    """
    Select the (num_gloss + extra) frames with the largest changes from change,
    while ensuring that any two selected frames are at least min_gap frames apart
    """
    extra = 3
    K = num_gloss + extra

    idx = np.argsort(change)[::-1]

    selected = []
    for i in idx:
        frame_id = i + 1  # change[i] corresponds to frame i+1

        if all(abs(frame_id - s) >= min_gap for s in selected):
            selected.append(frame_id)

        if len(selected) >= K:
            break

    return sorted(selected)


# ============================================================
# Main
# ============================================================

def main():
    csl_root = "/CSL-Daily/frames_512x512"
    pt_root = "PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/train"

    csl_output_root = "/prompts/CSL"
    pt_output_root = "/prompts/pT"

    os.makedirs(csl_output_root, exist_ok=True)
    os.makedirs(pt_output_root, exist_ok=True)

    datasets = [
        ("CSL", csl_root, csl_output_root),
        ("pT", pt_root, pt_output_root),
    ]

    for dataset_name, frame_root, output_root in datasets:
        if not os.path.isdir(frame_root):
            raise FileNotFoundError(f"Frame root not found: {frame_root}")

        video_dirs = sorted(
            d for d in glob(os.path.join(frame_root, "*"))
            if os.path.isdir(d)
        )

        print(
            f"\nProcessing {dataset_name}: "
            f"{len(video_dirs)} video directories found."
        )

        for video_idx, video_dir in enumerate(video_dirs, start=1):
            video_name = os.path.basename(video_dir)

            print(
                f"[{dataset_name}] "
                f"[{video_idx}/{len(video_dirs)}] "
                f"Processing {video_name} ..."
            )

            try:
                frame_paths = load_frames(video_dir)

                if len(frame_paths) < 2:
                    print(
                        f"Skipping {video_name}: "
                        f"only {len(frame_paths)} frame(s)."
                    )
                    continue

                features = extract_features(frame_paths)

                w = 5
                change = compute_change_score(features, w)

                output_file = os.path.join(
                    output_root,
                    f"{video_name}.txt"
                )

                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(f"(total {len(frame_paths)} frames in this video)\n")
                    f.write(f"(change window size: {w})\n")

                    for i, score in enumerate(change):
                        frame_path = frame_paths[i + 1]
                        frame_name = os.path.basename(frame_path)

                        f.write(
                            f"{i + 2}. "
                            f"{frame_name} "
                            f"{float(score):.8f}\n"
                        )

                print(f"Saved change scores to {output_file}")

            except Exception as e:
                print(
                    f"Error processing {video_name}: {e}"
                )

    print("\nAll video directories processed!")


if __name__ == "__main__":
    main()