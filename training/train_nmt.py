"""
NMT Training — IIT Bombay En-Hi Corpus → BiLSTM Seq2Seq
Usage: python training/train_nmt.py
Output: models/nmt/nmt_model.pt + models/nmt/vocab.pt
Tip: Run on Google Colab T4 GPU for 6x speedup.
"""
import os, sys, math, time, json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.vocab import (
    build_vocab, encode, load_iitb_corpus,
    tokenize_en, tokenize_hi,
    PAD_IDX, SOS_IDX, EOS_IDX
)
from models.nmt.seq2seq import build_model

# ── CONFIG ────────────────────────────────────────────────────────────────────
EN_PATH    = "data/iitb_corpus/IITB.en-hi.en"
HI_PATH    = "data/iitb_corpus/IITB.en-hi.hi"
MODEL_OUT  = "models/nmt/nmt_model.pt"
VOCAB_OUT  = "models/nmt/vocab.pt"
MAX_PAIRS  = 300000   # use 50000 for quick CPU test
EPOCHS     = 20
BATCH_SIZE = 128
LR         = 1e-3
CLIP       = 1.0
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── DATASET ───────────────────────────────────────────────────────────────────
class TranslationDataset(Dataset):
    def __init__(self, pairs, src_vocab, tgt_vocab):
        self.data = []
        for en, hi in pairs:
            src = encode(tokenize_en(en), src_vocab)
            tgt = encode(tokenize_hi(hi), tgt_vocab)
            if 2 <= len(src) <= 52 and 2 <= len(tgt) <= 52:
                self.data.append((
                    torch.tensor(src, dtype=torch.long),
                    torch.tensor(tgt, dtype=torch.long)
                ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_pad = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_pad = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
    return src_pad, tgt_pad

# ── TRAINING LOOP ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for src, tgt in loader:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        optimizer.zero_grad()
        output = model(src, tgt, teacher_forcing_ratio=0.5)
        # output: (batch, tgt_len, vocab) — skip <sos> at position 0
        output = output[:, 1:].reshape(-1, output.shape[-1])
        tgt    = tgt[:, 1:].reshape(-1)
        loss = criterion(output, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)


def eval_epoch(model, loader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            output = model(src, tgt, teacher_forcing_ratio=0.0)
            output = output[:, 1:].reshape(-1, output.shape[-1])
            tgt    = tgt[:, 1:].reshape(-1)
            epoch_loss += criterion(output, tgt).item()
    return epoch_loss / len(loader)


def train():
    if not os.path.isfile(EN_PATH) or not os.path.isfile(HI_PATH):
        print("IIT Bombay corpus not found.")
        print(f"  Expected: {EN_PATH}")
        print("  See DOWNLOAD_DATASETS.txt for download instructions.")
        return

    print("Loading corpus...")
    pairs = load_iitb_corpus(EN_PATH, HI_PATH, max_pairs=MAX_PAIRS)
    print(f"  {len(pairs)} sentence pairs loaded")

    # Split 90/10
    split = int(len(pairs) * 0.9)
    train_pairs, val_pairs = pairs[:split], pairs[split:]

    print("Building vocabularies...")
    src_vocab = build_vocab([e for e, _ in train_pairs], tokenize_en,
                             max_vocab=20000, min_freq=2)
    tgt_vocab = build_vocab([h for _, h in train_pairs], tokenize_hi,
                             max_vocab=20000, min_freq=2)
    src_inv = {v: k for k, v in src_vocab.items()}
    tgt_inv = {v: k for k, v in tgt_vocab.items()}
    print(f"  English vocab: {len(src_vocab):,} words")
    print(f"  Hindi vocab:   {len(tgt_vocab):,} words")

    torch.save({
        'src_vocab': src_vocab, 'tgt_vocab': tgt_vocab,
        'src_inv': src_inv,    'tgt_inv': tgt_inv,
        'sos': SOS_IDX, 'eos': EOS_IDX, 'pad': PAD_IDX
    }, VOCAB_OUT)

    train_ds = TranslationDataset(train_pairs, src_vocab, tgt_vocab)
    val_ds   = TranslationDataset(val_pairs,   src_vocab, tgt_vocab)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_fn, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          collate_fn=collate_fn, num_workers=0)

    model = build_model(len(src_vocab), len(tgt_vocab), DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")
    print(f"Training on: {DEVICE}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, min_lr=1e-5
    )
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    best_val_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_dl, optimizer, criterion)
        val_loss   = eval_epoch(model, val_dl, criterion)
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        train_ppl = math.exp(train_loss)
        val_ppl   = math.exp(val_loss)

        print(f"Epoch {epoch:02d}/{EPOCHS}  "
              f"train_loss={train_loss:.4f} (ppl={train_ppl:.1f})  "
              f"val_loss={val_loss:.4f} (ppl={val_ppl:.1f})  "
              f"time={elapsed:.0f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state': model.state_dict(),
                'src_vocab_size': len(src_vocab),
                'tgt_vocab_size': len(tgt_vocab),
                'epoch': epoch,
                'val_loss': val_loss,
            }, MODEL_OUT)
            print(f"  --> Saved best model")

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    print(f"Run evaluate_nmt.py next to get BLEU score.")


if __name__ == "__main__":
    train()
