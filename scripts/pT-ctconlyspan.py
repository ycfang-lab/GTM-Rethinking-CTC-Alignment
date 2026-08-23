## Based on the CTC loss function with forced alignment,
## Architecture: ResNet18 (pretrained weights) - 1D CNN (k5, p2, k5, p2) - BiLSTM (512 hidden units in each direction) - CTC, i.e., the CSLR baseline,
## Decoding uses a Viterbi dynamic-programming forced-alignment strategy: given the ground-truth gloss sequence, find the most probable forced-alignment path.
## num_workers=8; training one epoch takes about 0.4h.

import os
import csv
from glob import glob
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import torchvision.models as models

import json
import time

LOG_PATH = "/log/pT-ctc-span-21lclog.jsonl"

def log_event(data: dict):
    data["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

# =========================================================
# Paths
# =========================================================
TRAIN_FRAMES_DIR = "/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/train"
DEV_FRAMES_DIR = "/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/dev"

TRAIN_CSV = "/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.train.corpus.csv"
DEV_CSV = "/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.dev.corpus.csv"

# =========================================================
# Build gloss vocab + samples
# =========================================================
def build_gloss_vocab(csv_paths):
    gloss_set = set()
    samples = []

    for csv_path in csv_paths:
        split = "train" if "train.corpus.csv" in os.path.basename(csv_path) else "dev"

        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f, delimiter='|')
            for row in reader:
                video = row["name"]
                gloss_seq = row["orth"].strip().split()
                for g in gloss_seq:
                    gloss_set.add(g)
                samples.append((split, video, gloss_seq))

    gloss_list = sorted(gloss_set)
    gloss2id = {g: i + 1 for i, g in enumerate(gloss_list)}  # 0 is blank
    id2gloss = {i: g for g, i in gloss2id.items()}

    return samples, gloss2id, id2gloss


# =========================================================
# Dataset
# =========================================================
class PhoenixCTCDataset(Dataset):
    def __init__(self, frames_roots, samples, gloss2id):
        self.frames_roots = frames_roots  # {"train": ..., "dev": ...}
        self.samples = samples            # list of (split, video, gloss_seq)
        self.gloss2id = gloss2id

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.samples)

    def load_frames(self, split, video_name):
        frame_dir = os.path.join(self.frames_roots[split], video_name)
        paths = sorted(glob(os.path.join(frame_dir, "*.png")))
        assert len(paths) > 0

        frames = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            frames.append(self.transform(img))

        return torch.stack(frames, dim=0)

    def __getitem__(self, idx):
        split, video, gloss_seq = self.samples[idx]
        frames = self.load_frames(split, video)
        targets = torch.tensor([self.gloss2id[g] for g in gloss_seq], dtype=torch.long)

        return {
            "video": video,
            "frames": frames,
            "targets": targets,
            "gt_gloss_seq": gloss_seq,
            "input_length": frames.shape[0],
            "target_length": targets.shape[0]
        }


def ctc_collate_fn(batch):
    frames = [b["frames"] for b in batch]
    targets = [b["targets"] for b in batch]

    input_lengths = torch.tensor([b["input_length"] for b in batch])
    target_lengths = torch.tensor([b["target_length"] for b in batch])

    frames = torch.nn.utils.rnn.pad_sequence(frames, batch_first=True)
    targets = torch.cat(targets, dim=0)

    videos = [b["video"] for b in batch]
    gt_gloss_seqs = [b["gt_gloss_seq"] for b in batch]

    return frames, targets, input_lengths, target_lengths, videos, gt_gloss_seqs


# =========================================================
# Model
# =========================================================
class CTCModel(nn.Module):
    def __init__(self, num_gloss):
        super().__init__()
        backbone = models.resnet18(pretrained=True)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.temporal_conv = nn.Sequential(  # K5 → P2 → K5 → P2
            nn.Conv1d(
                in_channels=512,
                out_channels=1024,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2), 
            nn.Conv1d(
                in_channels=1024,
                out_channels=1024,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.lstm = nn.LSTM(
            input_size=1024,
            hidden_size=512,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Linear(1024, num_gloss + 1)

    def forward(self, frames):
        B, T, C, H, W = frames.shape
        frames = frames.view(B * T, C, H, W)
        feats = self.backbone(frames)
        feats = feats.view(B, T, -1)

        feats = feats.transpose(1, 2)
        feats = self.temporal_conv(feats)
        feats = feats.transpose(1, 2)

        feats, _ = self.lstm(feats)
        return self.fc(feats)

# def get_blank_bias(epoch: int):
#     if epoch <= 10:
#         # Strongly alleviate blank collapse (completely flatten the blank prior to unlock the gloss path)
#         return 7.0
#     elif epoch <= 20:
#         # 7 -> 5 gradually relax the prior
#         return 7.0 - (epoch - 10) * (2.0 / 10)
#     elif epoch <= 30:
#         # 5 -> 3 transition from "prior correction" to "data-driven"
#         return 5.0 - (epoch - 20) * (2.0 / 10)
#     elif epoch <= 40:
#         # 3 -> 1 gradually return to standard CTC
#         return 3.0 - (epoch - 30) * (2.0 / 10)
#     else:
#         # 1 -> 0 fully return to standard CTC
#         return max(0.0, 1.0 - (epoch - 40) * (1.0 / 10))


# =========================================================
# CTC Forced Alignment
# =========================================================
def _build_ctc_extended_targets(targets: torch.Tensor, blank: int = 0):
    # Insert blanks between targets and at the beginning and end
    L = targets.numel()
    ext = torch.full((2 * L + 1,), blank, dtype=torch.long, device=targets.device)
    ext[1::2] = targets
    return ext


def ctc_forced_align_viterbi(log_probs, targets, blank=0, blank_penalty=3.5, warm_frames=3): # Penalize blank (no penalty for the first and last 3 frames)
    T, C = log_probs.shape # T: number of frames
    device = log_probs.device

    ext = _build_ctc_extended_targets(targets, blank)
    S = ext.numel() # S: gloss index in ext; number of states in ext (2*L+1)

    neg_inf = -1e9
    dp = torch.full((T, S), neg_inf, device=device)
    # dp[t][s]: the highest cumulative score (sum of log probabilities) obtainable by reaching position s in ext at frame t
    bp = torch.zeros((T, S), dtype=torch.long, device=device)
    # bp[t][s]: records the previous state from which dp[t][s] came for backtracking the optimal path

    # -------- Initialization (t = 0) --------
    # Note: t=0 < warm_frames, so blank is not penalized
    dp[0, 0] = log_probs[0, ext[0]]
    if S > 1:
        dp[0, 1] = log_probs[0, ext[1]]
    # dp[0, 0] = log_probs[0, ext[0]] - (blank_penalty if ext[0] == blank else 0.0)
    # if S > 1:
    #     dp[0, 1] = log_probs[0, ext[1]]

    # -------- Dynamic programming --------
    for t in range(1, T):
        for s in range(S):
            # stay: transition from s to s
            best = dp[t - 1, s] # Continue emitting the same symbol at this frame (if ext[s] is A, then A spans multiple consecutive frames)
            prev = s

            # step: transition from s-1 to s
            if s - 1 >= 0 and dp[t - 1, s - 1] > best: # blank -> A means "A starts now"; A -> blank means "A ends here, with a blank inserted in between".
                best = dp[t - 1, s - 1]
                prev = s - 1

            # skip: transition from s-2 to s (there may be no blank between two glosses)
            if (
                s - 2 >= 0
                and ext[s] != blank
                and ext[s] != ext[s - 2]
                and dp[t - 1, s - 2] > best
            ):
                best = dp[t - 1, s - 2]
                prev = s - 2

            emit = log_probs[t, ext[s]]
            # -------- blank penalty --------
            if ext[s] == blank:
                if warm_frames <= t < T - warm_frames:
                    emit = emit - blank_penalty  # blank penalty
                # else: do not penalize the first or last warm_frames
            
            dp[t, s] = best + emit
            bp[t, s] = prev

    # -------- Final-state selection --------
    s = S - 1
    if S - 2 >= 0 and dp[T - 1, S - 2] > dp[T - 1, S - 1]:
        s = S - 2

    # -------- Backtracking --------
    states = torch.zeros(T, dtype=torch.long, device=device)
    for t in range(T - 1, -1, -1):
        states[t] = s
        s = bp[t, s]

    return ext[states]


def spans_from_frame_ids_allow_blank(
    frame_ids,
    id2gloss,
    blank=0,
    max_blank_gap=2,
):
    spans = []
    cur = None
    start = None
    blank_count = 0
    T = frame_ids.numel()

    for t in range(T):
        p = int(frame_ids[t])

        if p == blank:
            if cur is not None:
                blank_count += 1
                if blank_count > max_blank_gap:
                    end = t - blank_count
                    spans.append((id2gloss[cur], start, end))
                    cur = None
                    start = None
                    blank_count = 0
            continue

        blank_count = 0
        if cur is None:
            cur = p
            start = t
        elif p != cur:
            spans.append((id2gloss[cur], start, t - 1))
            cur = p
            start = t

    if cur is not None:
        spans.append((id2gloss[cur], start, T - 1))

    return spans


# =========================================================
# INFER + SAVE
# =========================================================
def run_infer_and_save(model, infer_loader, gloss2id, id2gloss, device, out_dir, epoch):
    os.makedirs(out_dir, exist_ok=True)

    model.eval()
    results = []

    total_frames = 0
    blank_frames = 0
    total_span_len = 0
    total_spans = 0
    total_gloss = 0

    with torch.no_grad():
        for idx, (frames, _, _, _, videos, gt_gloss_seqs) in enumerate(infer_loader):
            frames = frames.to(device)
            orig_T = frames.shape[1]  # Original number of video frames

            logits = model(frames)[0]
            log_probs = F.log_softmax(logits, dim=-1)
            target_ids = torch.tensor(
                [gloss2id[g] for g in gt_gloss_seqs[0]],
                dtype=torch.long,
                device=log_probs.device
            )
            
            # # ---- inference-only temporal smoothing----
            # probs = F.softmax(logits, dim=-1)
            # probs = F.avg_pool1d(
            #     probs.T.unsqueeze(0),
            #     kernel_size=3,
            #     stride=1,
            #     padding=1
            # ).squeeze(0).T
            # probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            # log_probs = torch.log(probs.clamp_min(1e-8))
            # target_ids = torch.tensor(
            #     [gloss2id[g] for g in gt_gloss_seqs[0]],
            #     dtype=torch.long,
            #     device=logits.device
            # )
            
            # -------- adaptive blank penalty --------
            # log_probs: [T, C]
            # blank id = 0
            blank_logp_mean = log_probs[:, 0].mean()
            # target_ids: gloss ID sequence of the current video (excluding blank)
            target_logp_mean = log_probs[:, target_ids].mean()
            adaptive_blank_penalty = max(4, blank_logp_mean - target_logp_mean - 2)

            frame_ids = ctc_forced_align_viterbi(log_probs, target_ids, blank=0, blank_penalty=float(adaptive_blank_penalty.item()), warm_frames=1)
            spans = spans_from_frame_ids_allow_blank(frame_ids, id2gloss, blank=0, max_blank_gap=0)

            # -------- statistics --------
            T = frame_ids.numel()
            total_frames += T
            blank_frames += int((frame_ids == 0).sum().item())
            total_spans += len(spans)
            total_gloss += len(gt_gloss_seqs[0])

            for _, s, e in spans:
                total_span_len += (e - s + 1)

            for g, s, e in spans:
                results.append({
                    "video": videos[0],
                    "gloss": g,
                    "start_frame": s * 4,
                    "end_frame": min(e * 4 + 3, orig_T - 1)
                })

    blank_ratio = blank_frames / max(total_frames, 1)
    avg_span_len = total_span_len / max(total_spans, 1)
    span_gloss_ratio = total_spans / max(total_gloss, 1)

    # print(
    #     f"[Span Stats][Epoch {epoch}] "
    #     f"blank_ratio={blank_ratio:.4f}, "
    #     f"avg_span_len={avg_span_len:.2f}, "
    #     f"span/gloss={span_gloss_ratio:.3f}"
    # )

    log_event({
        "phase": "eval_stats",
        "epoch": epoch,
        "blank_ratio": float(blank_ratio),
        "avg_span_len": float(avg_span_len),
        "span_gloss_ratio": float(span_gloss_ratio)
    })

    out_csv = os.path.join(out_dir, "pT-span.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["video", "gloss", "start_frame", "end_frame"]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved forced alignment to {out_csv}")


# =========================================================
# Main
# =========================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    samples, gloss2id, id2gloss = build_gloss_vocab([TRAIN_CSV, DEV_CSV])
    frames_roots = {
        "train": TRAIN_FRAMES_DIR,
        "dev": DEV_FRAMES_DIR
    }
    dataset = PhoenixCTCDataset(frames_roots, samples, gloss2id)

    train_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=8,
        collate_fn=ctc_collate_fn,
        pin_memory=True,
        persistent_workers=True,
    )

    infer_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=1,
        collate_fn=ctc_collate_fn
    )

    model = CTCModel(len(gloss2id)).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1.5e-4)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[20, 30, 40],
        gamma=0.2
    )

    num_epochs = 40
    accum_steps = 2
    SAVE_EPOCHS = {5, 10, 20, 25, 30, 35, 40}
    SAVE_MODEL_EPOCHS = {20, 30, 40}

    for epoch in range(num_epochs):
        # print(f"\nEpoch {epoch + 1}")
        model.train()

        accum_counter = 0
        optimizer.zero_grad()

        for step, (frames, targets, in_lens, tgt_lens, _, _) in enumerate(train_loader):
            frames = frames.to(device)
            targets = targets.to(device)
            
            in_lens = torch.clamp(in_lens // 4, min=1)
            logits = model(frames)
            # # ===== blank logit bias (training only) =====
            # beta = get_blank_bias(epoch + 1)  # epoch is 0-based
            # if beta > 0:
            #     logits[..., 0] -= beta
            # ===== logit diagnostics (no grad) =====
            with torch.no_grad():
                blank_logit = logits[..., 0].mean().item()
                gloss_logit = logits[..., 1:].mean().item()
                logit_gap = gloss_logit - blank_logit
            # ============================================
            log_probs = F.log_softmax(logits, -1).permute(1, 0, 2)

            loss = criterion(log_probs, targets, in_lens, tgt_lens)
            loss = loss / accum_steps
            loss.backward()

            accum_counter += 1

            if accum_counter % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                accum_counter = 0

            if step == len(train_loader) - 1 and accum_counter > 0:
                optimizer.step()
                optimizer.zero_grad()
                accum_counter = 0

            if step % 100 == 0:
                actual_loss = loss.item() * accum_steps
                # print(f"  step {step}/ epoch {epoch + 1}, loss = {actual_loss:.4f}")
                log_event({
                    "phase": "train",
                    "epoch": epoch + 1,
                    "step": step,
                    "loss": round(actual_loss, 5),
                    "blank_logit": round(blank_logit, 5),
                    "gloss_logit": round(gloss_logit, 5),
                    "logit_gap": round(logit_gap, 5),
                    # "beta": float(beta),
                })

        scheduler.step()

        if (epoch + 1) in SAVE_EPOCHS:
            out_dir = f"/pT_ctc_span_epoch{epoch + 1}"
            run_infer_and_save(
                model=model,
                infer_loader=infer_loader,
                gloss2id=gloss2id,
                id2gloss=id2gloss,
                device=device,
                out_dir=out_dir,
                epoch=epoch + 1
            )
            if (epoch + 1) in SAVE_MODEL_EPOCHS:
                model_path = os.path.join(out_dir, "pT-CTCspan_model_state.pth")
                torch.save(model.state_dict(), model_path)
                print(f"Saved model state_dict to {model_path}")

if __name__ == "__main__":
    main()