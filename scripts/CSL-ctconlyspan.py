## Use only CTC + greedy decoding to locate approximate frame positions

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


# =========================================================
# LOG
# =========================================================
LOG_PATH = "log/CSL-ctc-span-21lclog.jsonl"

def log_event(data: dict):
    data["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


# =========================================================
# Paths (CSL-Daily)
# =========================================================
FRAMES_DIR = "/CSL-Daily/frames_512x512"
CSV_PATH = "/CSL-Daily/sentence_label/video_map.csv"


# =========================================================
# Build gloss vocab
# =========================================================
def build_gloss_vocab(csv_path):
    gloss_set = set()
    samples = []

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f, delimiter='|')
        for row in reader:
            video = row["name"]
            gloss_seq = row["gloss"].strip().split()

            for g in gloss_seq:
                gloss_set.add(g)

            samples.append((video, gloss_seq))

    gloss_list = sorted(gloss_set)
    gloss2id = {g: i + 1 for i, g in enumerate(gloss_list)}  # 0 blank
    id2gloss = {i: g for g, i in gloss2id.items()}

    return samples, gloss2id, id2gloss


# =========================================================
# Dataset
# =========================================================
class CSLCTCDataset(Dataset):
    def __init__(self, frames_root, samples, gloss2id):
        self.frames_root = frames_root
        self.samples = samples
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

    def load_frames(self, video_name):
        frame_dir = os.path.join(self.frames_root, video_name)
        paths = sorted(glob(os.path.join(frame_dir, "*.png")))
        assert len(paths) > 0, f"No frames in {frame_dir}"

        frames = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            frames.append(self.transform(img))

        return torch.stack(frames, dim=0)

    def __getitem__(self, idx):
        video, gloss_seq = self.samples[idx]
        frames = self.load_frames(video)
        targets = torch.tensor(
            [self.gloss2id[g] for g in gloss_seq],
            dtype=torch.long
        )

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


# =========================================================
# CTC Forced Alignment
# =========================================================
def _build_ctc_extended_targets(targets, blank=0):
    L = targets.numel()
    ext = torch.full((2 * L + 1,), blank, dtype=torch.long, device=targets.device)
    ext[1::2] = targets
    return ext


def ctc_forced_align_viterbi(log_probs, targets, blank=0, blank_penalty=3.5, warm_frames=3):
    T, C = log_probs.shape
    device = log_probs.device

    ext = _build_ctc_extended_targets(targets, blank)
    S = ext.numel()

    neg_inf = -1e9
    dp = torch.full((T, S), neg_inf, device=device)
    bp = torch.zeros((T, S), dtype=torch.long, device=device)

    dp[0, 0] = log_probs[0, ext[0]]
    if S > 1:
        dp[0, 1] = log_probs[0, ext[1]]

    for t in range(1, T):
        for s in range(S):

            best = dp[t - 1, s]
            prev = s

            if s - 1 >= 0 and dp[t - 1, s - 1] > best:
                best = dp[t - 1, s - 1]
                prev = s - 1

            if (
                s - 2 >= 0
                and ext[s] != blank
                and ext[s] != ext[s - 2]
                and dp[t - 1, s - 2] > best
            ):
                best = dp[t - 1, s - 2]
                prev = s - 2

            emit = log_probs[t, ext[s]]

            if ext[s] == blank:
                if warm_frames <= t < T - warm_frames:
                    emit = emit - blank_penalty

            dp[t, s] = best + emit
            bp[t, s] = prev

    s = S - 1
    if S - 2 >= 0 and dp[T - 1, S - 2] > dp[T - 1, S - 1]:
        s = S - 2

    states = torch.zeros(T, dtype=torch.long, device=device)
    for t in range(T - 1, -1, -1):
        states[t] = s
        s = bp[t, s]

    return ext[states]


def spans_from_frame_ids_allow_blank(frame_ids, id2gloss, blank=0, max_blank_gap=2):
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

    with torch.no_grad():
        for frames, _, _, _, videos, gt_gloss_seqs in infer_loader:

            frames = frames.to(device)
            orig_T = frames.shape[1]

            logits = model(frames)[0]
            log_probs = F.log_softmax(logits, dim=-1)

            target_ids = torch.tensor(
                [gloss2id[g] for g in gt_gloss_seqs[0]],
                dtype=torch.long,
                device=log_probs.device
            )
            
            # -------- adaptive blank penalty --------
            # log_probs: [T, C]
            # blank id = 0
            blank_logp_mean = log_probs[:, 0].mean()
            # target_ids: gloss id sequence of the current video (excluding blank)
            target_logp_mean = log_probs[:, target_ids].mean()
            adaptive_blank_penalty = max(4, blank_logp_mean - target_logp_mean - 2)

            frame_ids = ctc_forced_align_viterbi(
                log_probs,
                target_ids,
                blank=0,
                # blank_penalty=0,
                blank_penalty=float(adaptive_blank_penalty.item()),
                warm_frames=1
            )

            spans = spans_from_frame_ids_allow_blank(
                frame_ids,
                id2gloss,
                blank=0,
                max_blank_gap=0
            )

            for g, s, e in spans:
                results.append({
                    "video": videos[0],
                    "gloss": g,
                    "start_frame": s * 4,
                    "end_frame": min(e * 4 + 3, orig_T - 1)
                })

    out_csv = os.path.join(out_dir, "CSL-span.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["video", "gloss", "start_frame", "end_frame"]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved forced alignment to {out_csv}")


# =========================================================
# MAIN
# =========================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    samples, gloss2id, id2gloss = build_gloss_vocab(CSV_PATH)
    dataset = CSLCTCDataset(FRAMES_DIR, samples, gloss2id)

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
    SAVE_EPOCHS = {5, 10, 20, 30, 40}

    for epoch in range(num_epochs):

        model.train()

        for frames, targets, in_lens, tgt_lens, _, _ in train_loader:

            frames = frames.to(device)
            targets = targets.to(device)

            in_lens = torch.clamp(in_lens // 4, min=1)

            logits = model(frames)
            log_probs = F.log_softmax(logits, -1).permute(1, 0, 2)

            loss = criterion(log_probs, targets, in_lens, tgt_lens)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        if (epoch + 1) in SAVE_EPOCHS:
            out_dir = f"/CSL_ctc_span_epoch{epoch + 1}"

            run_infer_and_save(
                model,
                infer_loader,
                gloss2id,
                id2gloss,
                device,
                out_dir,
                epoch + 1
            )


if __name__ == "__main__":
    main()