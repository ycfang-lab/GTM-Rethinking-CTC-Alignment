# Training one epoch takes 30min; one dev inference takes 2min; one test inference takes 25min
import os
import json
import csv
import time
import glob
from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import functional as TF

from ctcdecode import CTCBeamDecoder


# =========================
# CSL-Daily configs
# =========================

FRAMES_ROOT = "/CSL-Daily/frames_512x512"

SPLIT_FILE = "/CSL-Daily/sentence_label/split_1.txt"
VIDEO_MAP_CSV = "/CSL-Daily/sentence_label/video_map.csv"

LOG_PATH = "/CSL-cslr-ctc+beam.jsonl"

EPOCHS = 50
BATCH_SIZE = 2
NUM_WORKERS = 8

ACCUM_STEPS = 2
LOG_EVERY_STEPS = 200

SAVE_EPOCHS = {10, 20, 30, 40, 50}
SAVE_ROOT_TMPL = "/root/autodl-tmp/z-csl-data/CSL-cslr-ctc+beam_epoch{ep}"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LR = 1e-4
WEIGHT_DECAY = 1e-4


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
            nn.Conv1d(512, 1024, 5, padding=2),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(1024, 1024, 5, padding=2),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
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

        self.fc = nn.Linear(1024, num_gloss + 1)

    def masked_bn(self, inputs, lengths):
        B, T, C, H, W = inputs.shape
        valid_frames = []
        for i in range(B):
            valid_frames.append(inputs[i, :lengths[i]])

        cat_frames = torch.cat(valid_frames, dim=0)
        feats = self.backbone(cat_frames)

        out = torch.zeros(B, T, feats.shape[-1], device=feats.device)
        start = 0
        for i in range(B):
            l = lengths[i]
            out[i, :l] = feats[start:start + l]
            start += l

        return out

    def forward(self, frames, lengths):
        feats = self.masked_bn(frames, lengths)
        feats = feats.transpose(1, 2)
        feats = self.temporal_conv(feats)
        feats = feats.transpose(1, 2)

        lengths_ds = lengths // 2
        lengths_ds = lengths_ds // 2
        lengths_ds = torch.clamp(lengths_ds, min=1)

        packed = nn.utils.rnn.pack_padded_sequence(
            feats, lengths_ds.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_out, _ = self.lstm(packed)

        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True
        )

        logits = self.fc(out)
        return logits, lengths_ds


# =========================
# Data loading
# =========================

def read_split_file(path):
    mapping = {}
    with open(path, "r") as f:
        lines = [x.strip() for x in f.readlines() if x.strip()]
    for line in lines[1:]:
        name, split = line.split("|")
        mapping[name.strip()] = split.strip()
    return mapping


def parse_video_map(path):
    mapping = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            name = row["name"].strip()
            gloss = row["gloss"].strip()

            if " " in gloss:
                tokens = gloss.split(" ")
            elif "|" in gloss:
                tokens = gloss.split("|")
            else:
                tokens = [gloss]

            mapping[name] = [g for g in tokens if g]

    return mapping


@dataclass
class Sample:
    name: str
    frame_paths: List[str]
    gloss_ids: List[int]


class CSLDailyDataset(Dataset):
    def __init__(self, names, gloss_map, gloss2id, frames_root, downsample_factor: int = 4, filter_illegal: bool = True):
        self.names = names
        self.gloss_map = gloss_map
        self.gloss2id = gloss2id
        self.downsample_factor = downsample_factor
        self.frames_root = frames_root
        
        self.names = []

        removed = 0

        for name in names:
            folder = os.path.join(self.frames_root, name)
            if not os.path.exists(folder):
                continue

            png_files = glob.glob(os.path.join(folder, "*.jpg"))
            png_count = len(png_files)

            gloss_seq = self.gloss_map.get(name, [])
            gloss_len = len(gloss_seq)

            down_len = png_count // self.downsample_factor

            if filter_illegal and down_len < gloss_len:
                removed += 1
                print(
                    f"[FILTER] {name} | png={png_count} | "
                    f"down={down_len} | gloss={gloss_len}"
                )
                continue

            self.names.append(name)

        print(f"Dataset initialized. Kept: {len(self.names)} | Removed: {removed}")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        folder = os.path.join(FRAMES_ROOT, name)
        frame_paths = sorted(glob.glob(os.path.join(folder, "*.jpg")))

        if len(frame_paths) == 0:
            return None

        gloss_seq = self.gloss_map.get(name, [])
        gloss_ids = [self.gloss2id[g] for g in gloss_seq if g in self.gloss2id]

        return Sample(name, frame_paths, gloss_ids)

# =========================
# Utilities: parsing & WER
# =========================
def append_jsonl(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        
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
    Use the token peak timesteps returned by ctcdecode to obtain a stable span approximation.
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
# Collate
# =========================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def apply_video_train_transform(imgs: List[Image.Image]) -> torch.Tensor:
    """
    Apply consistent random augmentation to the entire video
    return: [T, C, 224, 224]
    """
    # 1 resize (consistent)
    imgs = [TF.resize(img, (256, 256)) for img in imgs]
    # 2 random crop parameters (sample only once)
    i, j, h, w = transforms.RandomCrop.get_params(imgs[0], (224, 224))
    # 3 random flip (sample only once)
    do_flip = torch.rand(1).item() < 0.5

    out = []
    for img in imgs:
        img = TF.crop(img, i, j, h, w)
        if do_flip:
            img = TF.hflip(img)
        img = TF.to_tensor(img)
        img = TF.normalize(img, IMAGENET_MEAN, IMAGENET_STD)
        out.append(img)

    return torch.stack(out, dim=0)


def apply_video_eval_transform(imgs: List[Image.Image]) -> torch.Tensor:
    """
    Use deterministic processing for dev/test
    return: [T, C, 224, 224]
    """

    out = []
    for img in imgs:
        img = TF.resize(img, (256, 256))
        img = TF.center_crop(img, 224)
        img = TF.to_tensor(img)
        img = TF.normalize(img, IMAGENET_MEAN, IMAGENET_STD)
        out.append(img)

    return torch.stack(out, dim=0)

def collate_fn(samples, is_train=True):
    samples = [s for s in samples if s is not None]
    if len(samples) == 0:
        return None

    names = [s.name for s in samples]

    frame_tensors = []
    orig_lengths = []
    gloss_ids_list = []

    for s in samples:
        pil_imgs = [Image.open(p).convert("RGB") for p in s.frame_paths]

        if is_train:
            frames = apply_video_train_transform(pil_imgs)
        else:
            frames = apply_video_eval_transform(pil_imgs)
        frame_tensors.append(frames)
        orig_lengths.append(frames.shape[0])
        gloss_ids_list.append(s.gloss_ids)

    B = len(samples)
    Tmax = max(orig_lengths)
    C, H, W = frame_tensors[0].shape[1:]

    frames_pad = torch.zeros((B, Tmax, C, H, W), dtype=frame_tensors[0].dtype)
    for i, ft in enumerate(frame_tensors):
        frames_pad[i, :ft.shape[0]] = ft

    lengths = torch.tensor(orig_lengths)

    ctc_targets = []
    target_lengths = []

    for ids in gloss_ids_list:
        target_lengths.append(len(ids))
        ctc_targets.extend(ids)

    return (
        frames_pad,
        lengths,
        torch.tensor(ctc_targets, dtype=torch.long),
        torch.tensor(target_lengths, dtype=torch.long),
        names,
        lengths
    )

# =========================
# Dev/Test WER eval
# =========================
@torch.no_grad()
def evaluate_loader_wer(model: nn.Module, loader: DataLoader, id2gloss: Dict[int, str], decoder: CTCBeamDecoder, blank_id: int = 0):
    model.eval()
    total_edits = 0
    total_ref_words = 0
    total_blank = 0
    total_valid_steps = 0

    for batch in loader:
        if batch is None:
            continue
        frames, lengths_out, ctc_targets, target_lengths, names, orig_lengths_for_span = batch
        frames = frames.to(DEVICE, non_blocking=True)
        lengths_out = lengths_out.to(DEVICE)
        logits, lengths_ds = model(frames, lengths_out)
        lengths_ds = torch.clamp(lengths_ds, min=1)
        B, Tprime, V = logits.shape
        preds = logits.argmax(dim=-1)
        for i in range(B):
            L = int(lengths_ds[i].item())
            if L > 0:
                pred_seq = preds[i, :L]
                total_blank += (pred_seq == blank_id).sum().item()
                total_valid_steps += L

        # ---- decoder.decode() ----
        beam_results, timesteps, out_lens = decode_with_ctcdecoder(
            decoder=decoder,
            logits=logits,
            lengths_out=lengths_ds,
            blank_id=blank_id
        )

        ptr = 0
        for i in range(B):
            L = int(lengths_ds[i].item())
            if L <= 0:
                continue

            out_len = int(out_lens[i, 0].item())
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
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    out_csv = os.path.join(save_dir, "test_predictions.csv")
    rows = [("name", "pred_gloss")]

    for batch in test_loader:
        if batch is None:
            continue
        # frames, frame_labels_ds, lengths_out, ctc_targets, target_lengths, names, orig_lengths_for_span = batch
        frames, lengths_out, targets, target_lengths, names, orig_lengths_for_span = batch
        frames = frames.to(DEVICE, non_blocking=True)
        # logits, feats = model(frames)  # [B, T', V]
        logits, lengths_ds = model(frames, lengths_out)
        B, Tprime, V = logits.shape
        # lengths_ds = lengths_out // 2 // 2
        lengths_ds = torch.clamp(lengths_ds, min=1)
        
        beam_results, timesteps, out_lens = decode_with_ctcdecoder(
            decoder=decoder,
            logits=logits,
            lengths_out=lengths_ds,
            blank_id=blank_id
        )

        for i in range(B):
            name = names[i]
            L = int(lengths_ds[i].item())

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
                    orig_T = orig_lengths_for_span[i].item()
                    L = lengths_ds[i].item()
                    ratio = orig_T / L
                    start_frame = int(s_t * ratio) + 1
                    end_frame = int((e_t + 1) * ratio)
                    formatted.append(f"{g}({start_frame}-{end_frame})")
                rows.append((name, " ".join(formatted)))
            else:
                hyp = [id2gloss[x] for x in hyp_ids_f]
                rows.append((name, " ".join(hyp)))

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    with open(os.path.join(save_dir, "id2gloss.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in id2gloss.items()}, f, ensure_ascii=False, indent=2)

    print(f"[Infer] saved predictions to: {out_csv}")

# =========================
# Main
# =========================

def main():

    split_map = read_split_file(SPLIT_FILE)
    gloss_map = parse_video_map(VIDEO_MAP_CSV)

    train_names = [n for n, s in split_map.items() if s == "train"]
    dev_names = [n for n, s in split_map.items() if s == "dev"]
    test_names = [n for n, s in split_map.items() if s == "test"]

    # vocab from train
    gloss_set = set()
    for n in train_names:
        for g in gloss_map.get(n, []):
            gloss_set.add(g)

    gloss_list = sorted(list(gloss_set))

    blank_id = 0
    gloss2id = {g: i + 1 for i, g in enumerate(gloss_list)}
    id2gloss = {v: k for k, v in gloss2id.items()}
    num_gloss = len(gloss2id)

    train_ds = CSLDailyDataset(
        train_names,
        gloss_map,
        gloss2id,
        frames_root=FRAMES_ROOT,
        downsample_factor=4,
        filter_illegal=True
    )

    dev_ds = CSLDailyDataset(
        dev_names,
        gloss_map,
        gloss2id,
        frames_root=FRAMES_ROOT,
        downsample_factor=4,
        filter_illegal=True
    )

    test_ds = CSLDailyDataset(
        test_names,
        gloss_map,
        gloss2id,
        frames_root=FRAMES_ROOT,
        downsample_factor=4,
        filter_illegal=True
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=lambda b: collate_fn(b, True),
        persistent_workers=True,
    )
    
    train_eval_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=lambda b: collate_fn(b, False),
        persistent_workers=True,
    )
    
    dev_loader = DataLoader(
        dev_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=lambda b: collate_fn(b, False),
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=1,
        collate_fn=lambda b: collate_fn(b, False),
    )

    model = CTCModel(num_gloss).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[20, 30, 40],
        gamma=0.2
    )
    ctc_loss_fn = nn.CTCLoss(blank=blank_id, reduction="none", zero_infinity=False)
    
    labels = ["<blk>"] + [id2gloss[i] for i in range(1, num_gloss + 1)]
    decoder = CTCBeamDecoder(
        labels,
        beam_width=10,
        blank_id=blank_id,
        num_processes=8,
        log_probs_input=True,   # Pass log_probs after log_softmax
    )
    
    global_step = 0
    update_step = 0
    best_dev_wer = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        accum_counter = 0

        for step, batch in enumerate(train_loader):
            if batch is None:
                continue

            frames, lengths, targets, target_lengths, names, _ = batch
            frames = frames.to(DEVICE)
            lengths = lengths.to(DEVICE)
            targets = targets.to(DEVICE)
            target_lengths = target_lengths.to(DEVICE)

            logits, lengths_ds = model(frames, lengths)
            lengths_ds = torch.clamp(lengths_ds, min=1)
            B, Tprime, V = logits.shape
            
            # ---- blank ratio (train) ----
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                mask_valid = torch.arange(Tprime, device=DEVICE).unsqueeze(0) < lengths_ds.unsqueeze(1)
                preds_valid = preds[mask_valid]
                blank_ratio_train = (
                    0.0 if preds_valid.numel() == 0
                    else (preds_valid == blank_id).float().mean().item()
                )

            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
            loss = ctc_loss_fn(
                log_probs,
                targets,
                lengths_ds,
                target_lengths
            ).mean()

            # grad accum
            (loss / ACCUM_STEPS).backward()
            accum_counter += 1

            if accum_counter % ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                accum_counter = 0
                update_step += 1

            if step == len(train_loader) - 1 and accum_counter > 0:
                optimizer.step()
                optimizer.zero_grad()
                accum_counter = 0
                update_step += 1

            if (global_step % LOG_EVERY_STEPS) == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                log_obj = {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "epoch": epoch + 1,
                    "step": step,
                    "lr": round(lr_now, 6),
                    "train-blank_ratio": float(blank_ratio_train),
                    "loss": float(loss.item())
                }
                append_jsonl(LOG_PATH, log_obj)
                print(json.dumps(log_obj, ensure_ascii=False))

            global_step += 1

        if (epoch + 1) % 5 == 0:
            # dev trainWER each 5 epoch
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
            
        if (epoch + 1) % 1 == 0:
            # dev WER each epoch
            dev_wer, dev_blank_ratio = evaluate_loader_wer(model, dev_loader, decoder=decoder, id2gloss=id2gloss, blank_id=blank_id)
            lr_now = optimizer.param_groups[0]["lr"]
            summary = {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "epoch": epoch + 1,
                "lr": lr_now,
                "dev_wer": float(dev_wer),
                "dev_blank_ratio": float(dev_blank_ratio),
            }
            append_jsonl(LOG_PATH, summary)
            print(json.dumps(summary, ensure_ascii=False))

            if dev_wer < best_dev_wer:
                best_dev_wer = dev_wer
            
            scheduler.step()

        if (epoch + 1) in SAVE_EPOCHS:
            save_dir = SAVE_ROOT_TMPL.format(ep=epoch + 1)
            os.makedirs(save_dir, exist_ok=True)

            torch.save(model.state_dict(), os.path.join(save_dir, "model_state_dict.pt"))
            with open(os.path.join(save_dir, "gloss2id.json"), "w", encoding="utf-8") as f:
                json.dump(gloss2id, f, ensure_ascii=False, indent=2)

            infer_test_and_save(model, test_loader, decoder=decoder, id2gloss=id2gloss, save_dir=save_dir, blank_id=blank_id, with_span=True)

            test_wer, test_blank_ratio = evaluate_loader_wer(model, test_loader, decoder=decoder, id2gloss=id2gloss, blank_id=blank_id)
            append_jsonl(LOG_PATH, {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "epoch": epoch + 1,
                "test_wer": float(test_wer),
                "test_blank_ratio": float(test_blank_ratio),
            })

    print(f"Training done. Best dev WER = {best_dev_wer:.4f}")


if __name__ == "__main__":
    main()