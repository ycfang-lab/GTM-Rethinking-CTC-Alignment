# Training one epoch takes 30min; one dev inference takes 2min; one test inference takes 25min
import os
import re
import json
import csv
import time
import glob
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from torchvision import models, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import functional as TF

from collections import defaultdict

from ctcdecode import CTCBeamDecoder


# =========================
# User configs
# =========================
SPLIT_FILE = "/CSL-Daily/sentence_label/split_1.txt"
FRAMES_ROOT = "/CSL-Daily/frames_512x512"
TRAINDEV_ANN = "csl_train_finalspan.txt"
TEST_MAP_CSV = "/CSL-Daily/sentence_label/video_map.csv"

LOG_PATH = "/CSL-cslr-ce+ctc-ta.jsonl"

EPOCHS = 60
BATCH_SIZE = 2
NUM_WORKERS = 8

ACCUM_STEPS = 2  # gradient accumulation steps
LOG_EVERY_STEPS = 200

SAVE_EPOCHS = {25, 30, 35, 40, 45, 50, 55, 60}
SAVE_ROOT_TMPL = "/CSL-cslr-ce+ctc-ta_epoch{ep}"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Optim / LR schedule
LR = 1e-4
WEIGHT_DECAY = 1e-4
# GRAD_CLIP_NORM = 5.0

# CE settings
LABEL_SMOOTHING = 0.1
BLANK_WEIGHT = 0.2  # lower weight for blank in CE to avoid blank dominance

# Span output settings
DOWNSAMPLE_FACTOR = 4  # due to 2x MaxPool1d stride2


# =========================
# DATA AUGMENTATION
# =========================

def temporal_rescale_with_labels(
    frames: torch.Tensor,           # [T, C, H, W]
    spans: List[Tuple[int, int]],   # Original annotation spans (1-based inclusive)
    scale: float                    # Scale factor
) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """
    Scale video frames and annotation spans synchronously
    
    Args:
        frames: [T, C, H, W]
        spans: [(start, end), ...] 1-based inclusive
        scale: Scale factor
    
    Returns:
        scaled_frames: [T_new, C, H, W]
        scaled_spans: Scaled annotation spans (1-based inclusive)
    """
    T = frames.shape[0]
    T_new = max(1, int(T * scale))
    
    # 1. Frame scaling: linear interpolation sampling
    indices = torch.linspace(0, T-1, T_new).round().long()
    scaled_frames = frames[indices]  # [T_new, C, H, W]
    
    # 2. Span scaling: map original indices to new indices
    # Original index range: 1 to T (1-based)
    # New index range: 1 to T_new (1-based)
    scaled_spans = []
    for start, end in spans:
        # Convert to 0-based indices
        start_0 = start - 1
        end_0 = end - 1
        
        # Map to new indices
        new_start_0 = start_0 * scale
        new_end_0 = end_0 * scale
        
        # Convert back to 1-based and keep within the valid range
        new_start = max(1, min(T_new, int(round(new_start_0)) + 1))
        new_end = max(1, min(T_new, int(round(new_end_0)) + 1))
        
        # Ensure start <= end
        if new_start > new_end:
            new_start, new_end = new_end, new_start
        
        scaled_spans.append((new_start, new_end))
        
    fixed_spans = []
    prev_end = 1

    for s, e in scaled_spans:
        s = max(s, prev_end)
        e = max(e, s)

        fixed_spans.append((s, e))
        prev_end = e + 1

    scaled_spans = fixed_spans
    
    return scaled_frames, scaled_spans

def apply_video_train_transform(imgs, spans=None, temporal_scale_range=(0.8, 1.2)):
    """
    Args:
        imgs: List[PIL.Image]
        spans: List of original annotation spans (optional)
        temporal_scale_range: Scaling range; (0.8, 1.2) means random scaling to 80%-120%
    
    Returns:
        frames: [T, C, H, W] augmented frames
        spans: Scaled annotation spans (if input spans is not None)
    """
    # 1. Spatial augmentation
    ## 1.1 resize
    imgs = [TF.resize(img, (256, 256)) for img in imgs]
    ## 1.2 Random crop (same parameters for the entire video)
    i, j, h, w = transforms.RandomCrop.get_params(imgs[0], (224, 224))
    ## 1.3 Random rotation
    do_flip = torch.rand(1).item() < 0.5
    frames = []
    for img in imgs:
        img = TF.crop(img, i, j, h, w)
        if do_flip:
            img = TF.hflip(img)
        img = TF.to_tensor(img)
        img = img * 2.0 - 1.0 # Normalization equivalent to /127.5 - 1
        frames.append(img)
    video = torch.stack(frames, dim=0)  # [T, C, H, W]
    # 2. Temporal scaling
    if spans is not None and len(spans) > 0:
        scale = torch.empty(1).uniform_(*temporal_scale_range).item()
        T = video.shape[0]
        gloss_len = len(spans)
        T_new = int(T * scale)
        T_ds = (T_new // 2) // 2
        # Core constraint
        min_required = gloss_len + 2
        if T_ds < min_required:
            # Skip augmentation
            return video, spans
        # Apply scaling
        video, scaled_spans = temporal_rescale_with_labels(video, spans, scale)
        return video, scaled_spans
    # When no annotations are available, return only the video and None as the second value
    return video, None


def apply_video_eval_transform(imgs):
    """
    return: [T, C, 224, 224]
    """
    out = []
    for img in imgs:
        img = TF.resize(img, (256, 256))
        img = TF.center_crop(img, 224)
        img = TF.to_tensor(img)
        # Normalize
        img = img * 2.0 - 1.0
        out.append(img)

    return torch.stack(out, dim=0)

# =========================
# Model
# =========================
class CTCModel(nn.Module):
    def __init__(self, num_gloss: int):
        super().__init__()

        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=5, padding=2),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.MaxPool1d(2),

            nn.Conv1d(1024, 1024, kernel_size=5, padding=2),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.MaxPool1d(2),
        )

        self.lstm = nn.LSTM(
            input_size=1024,
            hidden_size=512,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )

        self.mid_fc = nn.Linear(1024, num_gloss + 1)
        self.fc = nn.Linear(1024, num_gloss + 1)

    # ===============================
    # masked BN
    # ===============================
    def masked_bn(self, inputs, lengths):
        """
        inputs:  [B, T, C, H, W]
        lengths: [B]
        """

        B, T, C, H, W = inputs.shape

        valid_frames = []
        for i in range(B):
            valid_frames.append(inputs[i, :lengths[i]])

        cat_frames = torch.cat(valid_frames, dim=0)  # [sum_T, C, H, W]
        feats = self.backbone(cat_frames)            # [sum_T, 512]

        out = torch.zeros(B, T, feats.shape[-1], device=feats.device)

        start = 0
        for i in range(B):
            l = lengths[i]
            out[i, :l] = feats[start:start + l]
            start += l

        return out

    # ===============================
    # forward
    # ===============================
    def forward(self, frames: torch.Tensor, lengths: torch.Tensor):

        # ---- ResNet with masked BN ----
        feats = self.masked_bn(frames, lengths)   # [B, T, 512]

        # ---- TemporalConv ----
        feats = feats.transpose(1, 2)             # [B, 512, T]
        feats = self.temporal_conv(feats)         # [B, 1024, T']
        feats = feats.transpose(1, 2)             # [B, T', 1024]
        
        # mid_feats = feats

        # ---- downsample lengths ----
        lengths_ds = lengths // 2
        lengths_ds = lengths_ds // 2
        lengths_ds = torch.clamp(lengths_ds, min=1)

        # ---- pack ----
        packed = nn.utils.rnn.pack_padded_sequence(
            feats,
            lengths_ds.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_out, _ = self.lstm(packed)

        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True
        )  # [B, T', 1024]

        # ---- logits ----
        # mid_logits = self.mid_fc(feats)
        logits = self.fc(out)
        
        mid_logits = self.fc(feats)  # Intermediate supervision after 1D-CNN, sharing the classifier

        return mid_logits, logits, out, lengths_ds
        # return mid_logits, logits, mid_feats, out, lengths_ds

# =========================
# Utilities: parsing & WER
# =========================
def append_jsonl(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_split_file(path: str) -> Dict[str, str]:
    """Return name -> split (train/dev/test)"""
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    for ln in lines[1:]:
        parts = ln.split("|")
        if len(parts) != 2:
            continue
        name, split = parts[0].strip(), parts[1].strip()
        mapping[name] = split
    return mapping


_GLOSS_SPAN_RE = re.compile(r"\d+\]\s*([^\(\)\s]+)\s*\((\d+)\s*-\s*(\d+)\)")


def parse_train_dev_annotation(path: str) -> Dict[str, Dict]:
    """
    Parse training span annotation file
    Return dict: name -> {
        'gloss_seq': [gloss1, gloss2, ...],
        'spans': [(start, end), ...]  # 1-based inclusive
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    blocks = text.split("================================================================================")
    ann = {}

    for blk in blocks:
        blk = blk.strip()
        if not blk:
            continue

        m_name = re.search(r"name:\s*([A-Za-z0-9_]+)", blk)
        if not m_name:
            continue
        name = m_name.group(1).strip()

        glosses = []
        spans = []
        for m in _GLOSS_SPAN_RE.finditer(blk):
            g = m.group(1).strip()
            s = int(m.group(2))
            e = int(m.group(3))
            glosses.append(g)
            spans.append((s, e))

        if len(glosses) == 0:
            continue

        ann[name] = {"gloss_seq": glosses, "spans": spans}

    return ann


def parse_test_video_map(path: str) -> Dict[str, List[str]]:
    """
    Parse video_map.csv:
    index|name|length|gloss|char|word|postag
    Return name -> gloss_seq(list[str])
    """
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("name") or "").strip()
            gloss = (row.get("gloss") or "").strip()
            if not name:
                continue
            if not gloss:
                mapping[name] = []
                continue
            if " " in gloss:
                toks = [t for t in gloss.split(" ") if t]
            elif "|" in gloss:
                toks = [t for t in gloss.split("|") if t]
            elif "/" in gloss:
                toks = [t for t in gloss.split("/") if t]
            else:
                toks = [gloss]
            mapping[name] = toks
    return mapping


def edit_distance(ref: List[str], hyp: List[str]) -> int:
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )
    return dp[n][m]

# =========================
# Decoder helpers (CTCBeamDecoder)
# =========================
def decode_with_ctcdecoder(
    decoder: CTCBeamDecoder,
    logits: torch.Tensor,
    lengths_out: torch.Tensor,
    blank_id: int = 0
):
    """
    logits: [B, T, V] (raw logits)
    lengths_out: [B] valid lengths (T per sample)
    Return:
      beam_results: [B, N_BEAMS, T]
      timesteps:    [B, N_BEAMS, T]
      out_lens:     [B, N_BEAMS]
    """
    # ctcdecode expects CPU tensors; use log_probs_input=True in decoder init
    log_probs = logits.log_softmax(dim=-1).cpu().contiguous()
    lengths_cpu = lengths_out.cpu().contiguous()

    beam_results, beam_scores, timesteps, out_lens = decoder.decode(log_probs, seq_lens=lengths_cpu)
    return beam_results, timesteps, out_lens


def timesteps_to_spans(token_ids: List[int], token_times: List[int], T: int):
    """
    Use token peak timesteps returned by ctcdecode to obtain a stable span approximation.
    The length of token_times should equal len(token_ids).
    Return [(token_id, start_t, end_t)], where t is the downsampled time step (0-based, within [0, T-1]).
    """
    if len(token_ids) == 0:
        return []
    if len(token_times) != len(token_ids):
        # Fallback: if timesteps are unavailable, assign a point span to each token
        return [(tid, min(i, T - 1), min(i, T - 1)) for i, tid in enumerate(token_ids)]

    # Ensure monotonicity (timesteps should already be monotonic in theory, but enforce it for safety)
    times = [max(0, min(int(t), T - 1)) for t in token_times]
    for i in range(1, len(times)):
        if times[i] < times[i - 1]:
            times[i] = times[i - 1]

    # Split intervals using the midpoint between adjacent token times
    cuts = []
    for i in range(len(times) - 1):
        mid = (times[i] + times[i + 1]) // 2
        cuts.append(mid)

    spans = []
    for i, tid in enumerate(token_ids):
        start = 0 if i == 0 else (cuts[i - 1] + 1)
        end = (cuts[i] if i < len(cuts) else (T - 1))
        if end < start:
            end = start
        spans.append((tid, start, end))
    return spans

# =========================
# Frame label downsampling
# =========================

def build_frame_labels_from_spans(
    T: int,
    gloss_ids: List[int],
    spans_1based: List[Tuple[int, int]],
    blank_id: int = 0,
    ignore_index: int = -100,
    shrink_ratio: float = 0.3, # Default value; dynamically passed in collate_fn
):
    labels = [ignore_index] * T   # Ignore by default

    for gid, (s1, e1) in zip(gloss_ids, spans_1based):
        s0 = max(0, s1 - 1)
        e0 = min(T - 1, e1 - 1)
        if s0 > e0:
            continue

        length = e0 - s0 + 1
        margin = int(length * shrink_ratio)

        new_s = s0 + margin
        new_e = e0 - margin

        if new_s > new_e:
            continue

        for t in range(new_s, new_e + 1):
            labels[t] = gid

    return labels


def viterbi_ctc_alignment(logits: torch.Tensor,
                          target: torch.Tensor,
                          blank_id: int):
    """
    logits: [T', V]
    target: [U]
    return alignment: [T'] one label per frame (including blank)
    """
    T, V = logits.shape
    U = target.size(0)

    log_probs = F.log_softmax(logits, dim=-1)

    # Expand target [b,y1,b,y2,...,b]
    ext = torch.full((2 * U + 1,), blank_id,
                     dtype=torch.long, device=logits.device)
    ext[1::2] = target

    S = ext.size(0)
    dp = torch.full((T, S), -1e9, device=logits.device)
    ptr = torch.zeros((T, S), dtype=torch.long, device=logits.device)

    dp[0, 0] = log_probs[0, ext[0]]
    if S > 1:
        dp[0, 1] = log_probs[0, ext[1]]

    for t in range(1, T):
        for s in range(S):
            candidates = [dp[t - 1, s]]
            if s - 1 >= 0:
                candidates.append(dp[t - 1, s - 1])
            if s - 2 >= 0 and ext[s] != blank_id and ext[s] != ext[s - 2]:
                candidates.append(dp[t - 1, s - 2])

            best = torch.stack(candidates).max()
            dp[t, s] = best + log_probs[t, ext[s]]

    s = dp[-1].argmax()
    alignment = []
    for t in reversed(range(T)):
        alignment.append(ext[s].item())
        if t > 0:
            # Backtrack
            candidates = []
            if s >= 0:
                candidates.append((dp[t - 1, s], s))
            if s - 1 >= 0:
                candidates.append((dp[t - 1, s - 1], s - 1))
            if s - 2 >= 0:
                candidates.append((dp[t - 1, s - 2], s - 2))
            s = max(candidates)[1]

    return torch.tensor(alignment[::-1], device=logits.device)

def downsample_labels_by_factor4_majority(labels: List[int]) -> List[int]:
    """
    Downsample per-frame labels to match T' ~ T/4
    using majority vote on each group of 4 frames.
    """
    T = len(labels)
    T_out = (T // 2) // 2  # floor(floor(T/2)/2)

    out = []
    for i in range(T_out):
        g = labels[DOWNSAMPLE_FACTOR * i: DOWNSAMPLE_FACTOR * i + DOWNSAMPLE_FACTOR]
        counts = {}
        for x in g:
            counts[x] = counts.get(x, 0) + 1
        # tie-break: prefer non-blank
        sorted_items = sorted(counts.items(), key=lambda kv: (kv[1], kv[0] != 0), reverse=True)
        out.append(sorted_items[0][0])
    return out


# =========================
# Dataset
# =========================
@dataclass
class Sample:
    name: str
    frame_paths: List[str]
    gloss_ids: Optional[List[int]]
    spans: Optional[List[Tuple[int, int]]]  # 1-based inclusive


class CSLDailyCSLRDataset(Dataset):
    def __init__(
        self,
        names: List[str],
        frames_root: str,
        name2gloss_spans: Optional[Dict[str, Dict]],
        name2test_gloss: Optional[Dict[str, List[str]]],
        gloss2id: Dict[str, int],
        is_train_or_dev: bool,
        downsample_factor: int = 4,
        filter_illegal: bool = True
    ):
        self.names = names
        self.frames_root = frames_root
        self.name2gloss_spans = name2gloss_spans or {}
        self.name2test_gloss = name2test_gloss or {}
        self.gloss2id = gloss2id
        self.is_train_or_dev = is_train_or_dev
        self.current_shrink_ratio = 0.3  # Default value; updated in the training loop
        self.downsample_factor = downsample_factor
        
        self.names = []
        removed = 0

        for name in names:

            folder = os.path.join(self.frames_root, name)
            if not os.path.exists(folder):
                continue

            frame_paths = sorted(
                glob.glob(os.path.join(folder, "*.jpg")) +
                glob.glob(os.path.join(folder, "*.png"))
            )

            T = len(frame_paths)
            if T == 0:
                continue

            # ------------- Get gloss sequence length -------------
            if is_train_or_dev:
                info = self.name2gloss_spans.get(name, None)
                if info is None:
                    continue
                gloss_seq = info["gloss_seq"]
            else:
                gloss_seq = self.name2test_gloss.get(name, [])

            gloss_len = len(gloss_seq)

            # ------------- Check validity -------------
            T_ds = T // self.downsample_factor

            if filter_illegal and T_ds < gloss_len:
                removed += 1
                print(
                    f"[FILTER] {name} | frames={T} | "
                    f"T//{self.downsample_factor}={T_ds} | gloss={gloss_len}"
                )
                continue

            self.names.append(name)

        print(f"[Dataset] Kept: {len(self.names)} | Removed: {removed}")

    def __len__(self):
        return len(self.names)

    def _list_frames(self, name: str) -> List[str]:
        folder = os.path.join(self.frames_root, name)
        fps = sorted(glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png")))
        return fps

    def __getitem__(self, idx: int) -> Optional[Sample]:
        name = self.names[idx]
        frame_paths = self._list_frames(name)
        if len(frame_paths) == 0:
            print(f"[Warning] No frames for {name}, skip.")
            return None

        try:
            gloss_ids = None
            spans = None

            if self.is_train_or_dev:
                info = self.name2gloss_spans.get(name, None)
                if info is None:
                    print(f"[Warning] Missing annotation for {name}, skip.")
                    return None

                gloss_seq = info["gloss_seq"]
                spans = info["spans"]

                gloss_ids = []
                for g in gloss_seq:
                    if g not in self.gloss2id:
                        print(f"[Warning] OOV gloss {g} in {name}, skip.")
                        return None
                    gloss_ids.append(self.gloss2id[g])

            else:
                gloss_seq = self.name2test_gloss.get(name, [])
                gloss_ids = [self.gloss2id[g] for g in gloss_seq if g in self.gloss2id]
                spans = None

            return Sample(name=name, frame_paths=frame_paths,
                        gloss_ids=gloss_ids, spans=spans)

        except Exception as e:
            print(f"[Warning] Error loading {name}: {e}, skip.")
            return None


def collate_fn(samples: List[Sample], blank_id: int = 0, shrink_ratio: float = 0.3, is_train=True):
    """
    Returns:
      frames_pad: [B, Tmax, C, H, W]
      frame_labels_ds: [B, Lmax] (train only; otherwise blank)
      lengths_out: [B] (T' lengths)
      ctc_targets: concatenated 1D (train/dev/test)
      target_lengths: [B]
      names: [B]
    """
    samples = [s for s in samples if s is not None]
    if len(samples) == 0:
        return None
    
    names = [s.name for s in samples]

    frame_tensors = []
    orig_lengths = []
    gloss_ids_list = []
    spans_list = []

    for s in samples:
        pil_imgs = [Image.open(p).convert("RGB") for p in s.frame_paths]

        if is_train:
            if s.spans is not None:
                frames, scaled_spans = apply_video_train_transform(pil_imgs, spans=s.spans)
                # Update spans to the scaled spans
                spans_list.append(scaled_spans)
            else:
                frames = apply_video_train_transform(pil_imgs)
                spans_list.append(None)
        else:
            frames = apply_video_eval_transform(pil_imgs)
            spans_list.append(s.spans)  # No scaling during evaluation
        frame_tensors.append(frames)
        orig_lengths.append(frames.shape[0])
        gloss_ids_list.append(s.gloss_ids)
        # spans_list.append(s.spans)

    B = len(samples)
    Tmax = max(orig_lengths)
    C, H, W = frame_tensors[0].shape[1:]

    frames_pad = torch.zeros((B, Tmax, C, H, W), dtype=frame_tensors[0].dtype)
    for i, ft in enumerate(frame_tensors):
        t = ft.shape[0]
        frames_pad[i, :t] = ft

    # model output lengths (two stride-2 pools)
    lengths_out = [ (t // 2) // 2 for t in orig_lengths ]
    Lmax = max(lengths_out) if max(lengths_out) > 0 else 1
    lengths_out_t = torch.tensor(orig_lengths, dtype=torch.long)

    # frame labels downsampled for CE (only meaningful for train)
    frame_labels_ds = torch.full((B, Lmax), fill_value=-100, dtype=torch.long)

    # CTC targets
    ctc_targets: List[int] = []
    target_lengths: List[int] = []

    for i in range(B):
        t = orig_lengths[i]
        L = lengths_out[i]

        if spans_list[i] is not None and gloss_ids_list[i] is not None:
            # train: build frame labels from spans, then downsample
            per_frame = build_frame_labels_from_spans(
                T=t, gloss_ids=gloss_ids_list[i], spans_1based=spans_list[i], blank_id=blank_id, shrink_ratio=shrink_ratio  # Dynamically pass shrink_ratio
            )
            ds = downsample_labels_by_factor4_majority(per_frame)
            ds = ds[:L]
            if len(ds) < L:
                ds += [blank_id] * (L - len(ds))
            frame_labels_ds[i, :L] = torch.tensor(ds, dtype=torch.long)

        # train/dev/test: always provide sequence targets for WER/CTC if available
        if gloss_ids_list[i] is None:
            target_lengths.append(0)
        else:
            target_lengths.append(len(gloss_ids_list[i]))
            ctc_targets.extend(gloss_ids_list[i])

    ctc_targets_t = torch.tensor(ctc_targets, dtype=torch.long) if len(ctc_targets) > 0 else torch.zeros((0,), dtype=torch.long)
    target_lengths_t = torch.tensor(target_lengths, dtype=torch.long)

    return frames_pad, frame_labels_ds, lengths_out_t, ctc_targets_t, target_lengths_t, names


# =========================
# Dev/Test WER eval
# =========================
@torch.no_grad()
def evaluate_loader_wer(model: nn.Module, loader: DataLoader, id2gloss: Dict[int, str], decoder: CTCBeamDecoder, blank_id: int = 0) -> float:
    model.eval()
    total_edits = 0
    total_ref_words = 0
    total_blank = 0
    total_valid_steps = 0

    for batch in loader:
        if batch is None:
            continue
        frames, frame_labels_ds, lengths_out, ctc_targets, target_lengths, names = batch
        frames = frames.to(DEVICE, non_blocking=True)
        lengths_out = lengths_out.to(DEVICE, non_blocking=True)
        mid_logits, logits, feats, lengths_out = model(frames, lengths_out)
        B, Tprime, V = logits.shape
        preds = logits.argmax(dim=-1)  # [B, T']

        for i in range(B):
            L = int(lengths_out[i].item())
            if L > 0:
                # ---- blank ratio ----
                pred_seq = preds[i, :L]
                total_blank += (pred_seq == blank_id).sum().item()
                total_valid_steps += L

        # ---- decoder.decode() ----
        beam_results, timesteps, out_lens = decode_with_ctcdecoder(
            decoder=decoder,
            logits=logits,
            lengths_out=lengths_out,
            blank_id=blank_id
        )
        
        ptr = 0
        for i in range(B):
            L = int(lengths_out[i].item())
            if L <= 0:
                continue
            out_len = int(out_lens[i, 0].item())
            # ---- WER ----
            hyp_ids = beam_results[i, 0, :out_len].tolist()
            hyp = [id2gloss[x] for x in hyp_ids if (x != blank_id and x in id2gloss)]

            ref_len = int(target_lengths[i].item())
            ref_ids = ctc_targets[ptr: ptr + ref_len].tolist()
            ptr += ref_len
            ref = [id2gloss[x] for x in ref_ids if x in id2gloss]

            if len(ref) == 0:
                continue
            total_edits += edit_distance(ref, hyp)
            total_ref_words += len(ref)

    wer = 0.0 if total_ref_words == 0 else total_edits / float(total_ref_words)
    blank_ratio = 0.0 if total_valid_steps == 0 else total_blank / float(total_valid_steps)

    return wer, blank_ratio


# =========================
# Inference (save CSV)
# =========================
@torch.no_grad()
def infer_test_and_save(model: nn.Module, test_loader: DataLoader, decoder: CTCBeamDecoder, id2gloss: Dict[int, str],
                        save_dir: str, blank_id: int = 0, with_span: bool = True):
    """
    Saves:
      test_predictions.csv: name, pred_gloss
    If with_span=True, pred_gloss is:
      glossA(s-e) glossB(s-e)
    where s = start_t*4, e = end_t*4+3 (converted to 1-based: +1 .. +4)
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    out_csv = os.path.join(save_dir, "test_predictions.csv")
    rows = [("name", "pred_gloss")]

    for batch in test_loader:
        if batch is None:
            continue
        frames, frame_labels_ds, lengths_out, ctc_targets, target_lengths, names = batch
        frames = frames.to(DEVICE, non_blocking=True)
        lengths_out = lengths_out.to(DEVICE, non_blocking=True)
        mid_logits, logits, feats, lengths_out = model(frames, lengths_out)
        B, Tprime, V = logits.shape
        
        beam_results, timesteps, out_lens = decode_with_ctcdecoder(
            decoder=decoder,
            logits=logits,
            lengths_out=lengths_out,
            blank_id=blank_id
        )

        for i in range(B):
            name = names[i]
            L = int(lengths_out[i].item())

            if L <= 0:
                rows.append((name, ""))
                continue
            
            out_len = int(out_lens[i, 0].item())
            hyp_ids = beam_results[i, 0, :out_len].tolist()
            hyp_times = timesteps[i, 0, :out_len].tolist()

            # Safe filtering
            hyp_pairs = [(tid, tt) for tid, tt in zip(hyp_ids, hyp_times) if tid != blank_id and tid in id2gloss]
            hyp_ids_f = [p[0] for p in hyp_pairs]
            hyp_times_f = [p[1] for p in hyp_pairs]

            if with_span:
                segs = timesteps_to_spans(hyp_ids_f, hyp_times_f, T=L)
                formatted = []
                for gid, s_t, e_t in segs:
                    if gid not in id2gloss:
                        continue
                    g = id2gloss[gid]
                    start_frame = s_t * DOWNSAMPLE_FACTOR + 1
                    end_frame = e_t * DOWNSAMPLE_FACTOR + DOWNSAMPLE_FACTOR
                    formatted.append(f"{g}({start_frame}-{end_frame})")
                rows.append((name, " ".join(formatted)))
            else:
                # hyp_ids = greedy_ctc_decode(logits[i, :L].cpu(), blank_id=blank_id)
                hyp = [id2gloss[x] for x in hyp_ids_f]
                rows.append((name, " ".join(hyp)))

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    with open(os.path.join(save_dir, "id2gloss.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in id2gloss.items()}, f, ensure_ascii=False, indent=2)

    print(f"[Infer] saved predictions to: {out_csv}")


def main():
    # 1) Read split map
    name2split = read_split_file(SPLIT_FILE)
    train_names = [n for n, sp in name2split.items() if sp == "train"]
    dev_names = [n for n, sp in name2split.items() if sp == "dev"]
    test_names = [n for n, sp in name2split.items() if sp == "test"]
    print(f"train={len(train_names)} dev={len(dev_names)} test={len(test_names)}")

    # 2) Parse annotations
    train_dev_ann = parse_train_dev_annotation(TRAINDEV_ANN)
    test_map = parse_test_video_map(TEST_MAP_CSV)

    # 3) Build vocab from train
    gloss_set = set()
    for n in train_names:
        info = train_dev_ann.get(n, None)
        if info is None:
            continue
        for g in info["gloss_seq"]:
            gloss_set.add(g)
    gloss_list = sorted(list(gloss_set))

    blank_id = 0
    gloss2id = {g: (i + 1) for i, g in enumerate(gloss_list)}  # 1..V
    id2gloss = {v: k for k, v in gloss2id.items()}
    num_gloss = len(gloss2id)
    print(f"num_gloss(train)={num_gloss} (+blank => {num_gloss+1} classes)")

    # 4) Datasets / Loaders
    train_ds = CSLDailyCSLRDataset(
        names=train_names,
        frames_root=FRAMES_ROOT,
        name2gloss_spans=train_dev_ann,
        name2test_gloss=None,
        gloss2id=gloss2id,
        is_train_or_dev=True,
        downsample_factor=4,
        filter_illegal=True
    )
    dev_ds = CSLDailyCSLRDataset(
        names=dev_names,
        frames_root=FRAMES_ROOT,
        name2gloss_spans=None,
        name2test_gloss=test_map,
        gloss2id=gloss2id,
        is_train_or_dev=False,
        downsample_factor=4,
        filter_illegal=True
    )
    test_ds = CSLDailyCSLRDataset(
        names=test_names,
        frames_root=FRAMES_ROOT,
        name2gloss_spans=None,
        name2test_gloss=test_map,
        gloss2id=gloss2id,
        is_train_or_dev=False,
        downsample_factor=4,
        filter_illegal=True
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        collate_fn=lambda b: collate_fn(b, blank_id=blank_id, shrink_ratio=train_ds.current_shrink_ratio, is_train=True),
        persistent_workers=True,
    )
    train_eval_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        collate_fn=lambda b: collate_fn(b, blank_id=blank_id, shrink_ratio=0.3, is_train=False),
        persistent_workers=True,
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        collate_fn=lambda b: collate_fn(b, blank_id=blank_id, shrink_ratio=0.3, is_train=False),
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=1, pin_memory=True,
        collate_fn=lambda b: collate_fn(b, blank_id=blank_id, shrink_ratio=0.3, is_train=False),
    )

    # 5) Model / Optim / Scheduler
    model = CTCModel(num_gloss=num_gloss).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # Scheduler based on UPDATE steps (not dataloader steps)
    total_update_steps = (EPOCHS * len(train_loader) + ACCUM_STEPS - 1) // ACCUM_STEPS
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[20, 30, 40, 55],
        gamma=0.2
    )
    
    labels = ["<blk>"] + [id2gloss[i] for i in range(1, num_gloss + 1)]
    decoder = CTCBeamDecoder(
        labels,
        beam_width=10,
        blank_id=blank_id,
        num_processes=8,
        log_probs_input=True,   # Pass log_probs after log_softmax
    )

    # ctc_loss_fn = nn.CTCLoss(blank=blank_id, zero_infinity=True)
    ctc_loss_fn = nn.CTCLoss(blank=blank_id, reduction="none", zero_infinity=False)

    ce_weight = torch.ones(num_gloss + 1, dtype=torch.float, device=DEVICE)
    ce_weight[blank_id] = BLANK_WEIGHT

    global_step = 0  # dataloader steps
    update_step = 0  # optimizer steps

    best_dev_wer = float("inf")

    for epoch in range(EPOCHS):
        # if epoch == 10:
        #     # print(">>> Freezing mid_fc after epoch 10")
        #     for p in model.mid_fc.parameters():
        #         p.requires_grad = False
        # ===== Dynamic strategy adjustment =====
        # # 1. Dynamic shrink_ratio        
        if epoch < 5:
            current_shrink_ratio = 0.2
        elif epoch < 10:
            current_shrink_ratio = 0.15
        else:
            current_shrink_ratio = 0.1
        # Update shrink_ratio in the dataset for collate_fn
        train_ds.current_shrink_ratio = current_shrink_ratio
        
        # 2. dynamic lambda for CTC and CE        
        if epoch < 9:
            lambda_ce = 1.0
            lambda_ctc = 0.3

        elif epoch <= 20:
            t = (epoch - 9) / 11.0  # 20-9=11 epochs of transition
            lambda_ce = 1.0 * (1 - t) + 0.2 * t  # CE decreases linearly from 1.0 to 0.2
            lambda_ctc = 0.3 * (1 - t) + 1.0 * t  # CTC increases linearly from 0.3 to 1.0

        else:
            lambda_ce = 0.2
            lambda_ctc = 1.0
        
        model.train()
        epoch_start = time.time()

        accum_counter = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            if batch is None:
                continue
            frames, frame_labels_ds, lengths_out, ctc_targets, target_lengths, names = batch
            frames = frames.to(DEVICE, non_blocking=True)
            frame_labels_ds = frame_labels_ds.to(DEVICE, non_blocking=True)
            lengths_out = lengths_out.to(DEVICE, non_blocking=True)
            ctc_targets = ctc_targets.to(DEVICE, non_blocking=True)
            target_lengths = target_lengths.to(DEVICE, non_blocking=True)

            # mid_logits, logits, feats = model(frames)  # [B, T', V]
            mid_logits, logits, feats, lengths_out = model(frames, lengths_out)
            # mid_logits, logits, midfeats, feats, lengths_out = model(frames, lengths_out)
            B, Tprime, V = logits.shape
            
            # ---- blank ratio (train) ----
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                mask_valid = torch.arange(Tprime, device=DEVICE).unsqueeze(0) < lengths_out.unsqueeze(1)
                preds_valid = preds[mask_valid]
                blank_ratio_train = (
                    0.0 if preds_valid.numel() == 0
                    else (preds_valid == blank_id).float().mean().item()
                )

            # ----- Frame CE -----
            mask = torch.arange(Tprime, device=DEVICE).unsqueeze(0) < lengths_out.unsqueeze(1)  # [B, T']
            logits_flat = mid_logits[mask]
            labels_flat = frame_labels_ds[mask]  # [N]

            if logits_flat.numel() == 0:
                loss_frame = torch.tensor(0.0, device=DEVICE)
            else:
                loss_frame = F.cross_entropy(
                    logits_flat,
                    labels_flat,
                    weight=ce_weight,
                    ignore_index=-100,
                    reduction="mean",
                    label_smoothing=LABEL_SMOOTHING
                )
            
            log_probs_mid = mid_logits.log_softmax(dim=-1).transpose(0,1)
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)  # [T', B, V]
            
            loss_ctc = ctc_loss_fn(
                log_probs,
                ctc_targets,
                lengths_out,
                target_lengths
            ).mean()

            loss = lambda_ce * loss_frame + lambda_ctc * loss_ctc 

            # gradient accumulation
            loss_scaled = loss / ACCUM_STEPS
            loss_scaled.backward()
            accum_counter += 1

            # step on full accumulation
            if accum_counter % ACCUM_STEPS == 0:
                # torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
                optimizer.zero_grad()
                accum_counter = 0

                # scheduler.step()
                update_step += 1

            # handle last partial accumulation
            if step == len(train_loader) - 1 and accum_counter > 0:
                # torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
                optimizer.zero_grad()
                accum_counter = 0

                # scheduler.step()
                update_step += 1

            # logging every 200 dataloader steps
            if (global_step % LOG_EVERY_STEPS) == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                log_obj = {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "epoch": epoch + 1,
                    "step": step,
                    "lr": lr_now,
                    "train-blank_ratio": float(blank_ratio_train),
                    "shrink_ratio": current_shrink_ratio,
                    "lambda_ctc": float(lambda_ctc),
                    "loss_total": float(loss.item()),
                    "loss_frame": float(loss_frame.item()),
                    # "loss_ctc_mid": float(loss_ctc_mid.item()),
                    "loss_ctc": float(loss_ctc.item()),
                    # "loss_va": float(loss_va.item())
                }
                append_jsonl(LOG_PATH, log_obj)
                print(json.dumps(log_obj, ensure_ascii=False))

            global_step += 1

        # dev trainWER each 5 epoch
        if (epoch + 1) % 5 == 0:
            train_wer, train_blank_ratio = evaluate_loader_wer(
                model,
                train_eval_loader,
                id2gloss=id2gloss,
                decoder=decoder,
                blank_id=blank_id
            )
            lr_now = optimizer.param_groups[0]["lr"]
            train_summary = {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "epoch": epoch + 1,
                "lr": lr_now,
                "train_wer": float(train_wer),
                "train_blank_ratio": float(train_blank_ratio),
            }

            append_jsonl(LOG_PATH, train_summary)
            print(json.dumps(train_summary, ensure_ascii=False))
        
        dev_wer, dev_blank_ratio = evaluate_loader_wer(model, dev_loader, id2gloss=id2gloss, decoder=decoder, blank_id=blank_id)
        # epoch_time = time.time() - epoch_start
        lr_now = optimizer.param_groups[0]["lr"]

        summary = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": epoch + 1,
            "lr": lr_now,
            "dev_wer": float(dev_wer),
            "dev_blank_ratio": float(dev_blank_ratio),
        }
        append_jsonl(LOG_PATH, summary)
        # print(f"[Epoch {epoch+1}/{EPOCHS}] dev_WER={dev_wer:.4f} lr={lr_now:.6g} time={epoch_time:.1f}s")

        if dev_wer < best_dev_wer:
            best_dev_wer = dev_wer

        # inference + save at epoch 10/20/30/40
        if (epoch + 1) in SAVE_EPOCHS:
            save_dir = SAVE_ROOT_TMPL.format(ep=epoch + 1)
            os.makedirs(save_dir, exist_ok=True)

            # save state dict
            torch.save(model.state_dict(), os.path.join(save_dir, "model_state_dict.pt"))

            # save vocab
            with open(os.path.join(save_dir, "gloss2id.json"), "w", encoding="utf-8") as f:
                json.dump(gloss2id, f, ensure_ascii=False, indent=2)

            # infer on test & save csv with spans
            infer_test_and_save(model, test_loader, decoder=decoder, id2gloss=id2gloss, save_dir=save_dir, blank_id=blank_id, with_span=True)

            # compute test WER (NOTE: only for tokens in training vocab)
            test_wer, test_blank_ratio = evaluate_loader_wer(model, test_loader, id2gloss=id2gloss, decoder=decoder, blank_id=blank_id)
            append_jsonl(LOG_PATH, {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "epoch": epoch + 1,
                "test_wer": float(test_wer),
                "test_blank_ratio": float(test_blank_ratio),
            })

            # print(f"[Save] epoch={epoch+1} saved to {save_dir} | test_WER={test_wer:.4f}")

        scheduler.step()
        
    print(f"Training done. Best dev WER = {best_dev_wer:.4f}")


if __name__ == "__main__":
    main()