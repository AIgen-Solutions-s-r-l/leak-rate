"""Fine-tune a small Italian encoder for token classification on the train split (METHODS.md §9, `finetuned`).

Plain PyTorch loop (no accelerate dependency). BIO tags over the 16 identifier categories; documents are
tokenized with overflow windows (max_len tokens, stride overlap). CPU is enough for the synthetic corpus.

    python scripts/train_finetuned.py --model dbmdz/bert-base-italian-cased --epochs 2 --out models/finetuned-it
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from leakrate.schema import load_documents  # noqa: E402

CATS = ["PER", "CF", "IBAN", "PIVA", "EMAIL", "PHONE", "ADDRESS", "DOC_ID", "PLATE", "DATE_BIRTH",
        "ORG", "LOC", "DATE", "AGE", "PROFESSION", "HEALTH"]
LABELS = ["O"] + [f"{p}-{c}" for c in CATS for p in ("B", "I")]
L2I = {l: i for i, l in enumerate(LABELS)}


def char_labels(doc) -> list[str]:
    lab = ["O"] * len(doc.text)
    for e in sorted(doc.entities, key=lambda e: (e.start, -(e.end - e.start))):
        if any(lab[i] != "O" for i in range(e.start, e.end)):
            continue  # keep the first (longest-first within same start) annotation on overlap
        lab[e.start] = f"B-{e.category}"
        for i in range(e.start + 1, e.end):
            lab[i] = f"I-{e.category}"
    return lab


def encode(tokenizer, docs, max_len: int, stride: int):
    feats = []
    for d in docs:
        cl = char_labels(d)
        enc = tokenizer(d.text, return_offsets_mapping=True, truncation=True, max_length=max_len, stride=stride,
                        return_overflowing_tokens=True)
        for ids, offs in zip(enc["input_ids"], enc["offset_mapping"]):
            y = []
            prev_cat = None
            for (s, t) in offs:
                if s == t:  # special tokens
                    y.append(-100)
                    prev_cat = None
                    continue
                c = cl[s]
                if c == "O":
                    y.append(L2I["O"])
                    prev_cat = None
                else:
                    cat = c[2:]
                    tag = "B" if (c.startswith("B-") or prev_cat != cat) else "I"
                    y.append(L2I[f"{tag}-{cat}"])
                    prev_cat = cat
            feats.append((ids, y))
    return feats


class DS(Dataset):
    def __init__(self, feats):
        self.f = feats

    def __len__(self):
        return len(self.f)

    def __getitem__(self, i):
        return self.f[i]


def collate(batch, pad_id: int):
    n = max(len(ids) for ids, _ in batch)
    X = torch.full((len(batch), n), pad_id, dtype=torch.long)
    Y = torch.full((len(batch), n), -100, dtype=torch.long)
    M = torch.zeros((len(batch), n), dtype=torch.long)
    for i, (ids, y) in enumerate(batch):
        X[i, :len(ids)] = torch.tensor(ids)
        Y[i, :len(y)] = torch.tensor(y)
        M[i, :len(ids)] = 1
    return X, M, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/benchmark.jsonl")
    ap.add_argument("--model", default="dbmdz/bert-base-italian-cased")
    ap.add_argument("--out", default="models/finetuned-it")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="train docs (debug)")
    ap.add_argument("--domains", default="", help="comma-separated train domains (cross-template folds); default all")
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    torch.set_num_threads(max(1, torch.get_num_threads()))
    tok = AutoTokenizer.from_pretrained(a.model)
    docs = load_documents(a.data)
    doms = {x.strip() for x in a.domains.split(",") if x.strip()}
    train = [d for d in docs if d.split == "train" and (not doms or d.domain in doms)]
    dev = [d for d in docs if d.split == "dev" and (not doms or d.domain in doms)]
    if a.limit:
        train = train[:a.limit]; dev = dev[: max(10, a.limit // 10)]
    tf, df = encode(tok, train, a.max_len, a.stride), encode(tok, dev, a.max_len, a.stride)
    print(f"train docs {len(train)} -> {len(tf)} windows; dev docs {len(dev)} -> {len(df)} windows; labels {len(LABELS)}", flush=True)
    model = AutoModelForTokenClassification.from_pretrained(a.model, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)),
                                                            label2id=L2I)
    pad = tok.pad_token_id
    dl = DataLoader(DS(tf), batch_size=a.batch, shuffle=True, collate_fn=lambda b: collate(b, pad))
    ddl = DataLoader(DS(df), batch_size=a.batch, shuffle=False, collate_fn=lambda b: collate(b, pad))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    total = a.epochs * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, int(0.06 * total))) * max(0.0, 1 - s / total))
    t0 = time.time(); step = 0
    for ep in range(a.epochs):
        model.train()
        for X, M, Y in dl:
            out = model(input_ids=X, attention_mask=M, labels=Y)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if step % 20 == 0:
                print(f"  ep {ep + 1} step {step}/{total} loss {out.loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
        # dev token accuracy on labelled tokens (quick sanity; the real metric is the leak rate on the test split)
        model.eval(); hit = n = 0
        with torch.no_grad():
            for X, M, Y in ddl:
                pred = model(input_ids=X, attention_mask=M).logits.argmax(-1)
                mask = Y != -100
                hit += int(((pred == Y) & mask).sum()); n += int(mask.sum())
        print(f"epoch {ep + 1}: dev token accuracy {hit / max(n, 1):.4f} ({time.time() - t0:.0f}s)", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out); tok.save_pretrained(a.out)
    (Path(a.out) / "train_info.json").write_text(json.dumps({"base_model": a.model, "epochs": a.epochs, "batch": a.batch, "lr": a.lr,
                                                             "max_len": a.max_len, "stride": a.stride, "train_docs": len(train), "domains": sorted(doms),
                                                             "windows": len(tf), "seconds": time.time() - t0}, indent=1), encoding="utf-8")
    print(f"saved to {a.out}")


if __name__ == "__main__":
    main()
