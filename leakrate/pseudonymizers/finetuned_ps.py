"""Fine-tuned token-classification encoder (trained by scripts/train_finetuned.py) as a local pseudonymizer.

Reported as the upper bound for a local model on this benchmark: it has seen the train templates'
*families* (not the test templates) and disjoint name/city/organisation lists.
"""
from __future__ import annotations

import os

from ..schema import Span
from .base import Pseudonymizer
from .regex_ps import RegexPseudonymizer

DEFAULT_DIR = os.environ.get("LEAKRATE_FINETUNED_DIR", "models/finetuned-it")


class FinetunedPseudonymizer(Pseudonymizer):
    def __init__(self, with_regex: bool = True, model_dir: str = DEFAULT_DIR, max_len: int = 256, stride: int = 32, **kw):
        super().__init__(**kw)
        base = os.environ.get("LEAKRATE_FINETUNED_NAME", "finetuned")
        self.name = f"{base}+regex" if with_regex else base
        self._regex = RegexPseudonymizer(**kw) if with_regex else None
        import torch
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir).eval()
        self.id2label = self.model.config.id2label
        self.max_len, self.stride = max_len, stride

    def detect(self, text: str) -> list[Span]:
        enc = self.tok(text, return_offsets_mapping=True, truncation=True, max_length=self.max_len, stride=self.stride,
                       return_overflowing_tokens=True, return_tensors="pt", padding=True)
        with self.torch.no_grad():
            logits = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
        probs = logits.softmax(-1)
        # char-level vote: for overlapping windows keep the highest-probability label per character
        best: dict[int, tuple[float, str]] = {}
        for w in range(enc["input_ids"].shape[0]):
            for t, (s, e) in enumerate(enc["offset_mapping"][w].tolist()):
                if s == e:
                    continue
                p, i = probs[w, t].max(-1)
                lab = self.id2label[int(i)]
                for c in range(s, e):
                    if c not in best or best[c][0] < float(p):
                        best[c] = (float(p), lab)
        spans: list[Span] = []
        cur = None  # [start, end, cat, score_sum, n]
        for c in range(len(text)):
            lab = best.get(c, (0.0, "O"))[1]
            if lab == "O":
                if cur:
                    spans.append(Span(cur[0], cur[1], cur[2], cur[3] / cur[4], text[cur[0]:cur[1]])); cur = None
                continue
            tag, cat = lab.split("-", 1)
            if cur and cur[2] == cat and (tag == "I" or c == cur[1]) and tag != "B":
                cur[1] = c + 1; cur[3] += best[c][0]; cur[4] += 1
            elif cur and cur[2] == cat and tag == "B" and c == cur[1] and text[c - 1] not in " \n":
                cur[1] = c + 1; cur[3] += best[c][0]; cur[4] += 1  # B inside a word: continue
            else:
                if cur:
                    spans.append(Span(cur[0], cur[1], cur[2], cur[3] / cur[4], text[cur[0]:cur[1]]))
                cur = [c, c + 1, cat, best[c][0], 1]
        if cur:
            spans.append(Span(cur[0], cur[1], cur[2], cur[3] / cur[4], text[cur[0]:cur[1]]))
        # trim whitespace
        out = []
        for s in spans:
            a, b = s.start, s.end
            while a < b and text[a].isspace():
                a += 1
            while b > a and text[b - 1].isspace():
                b -= 1
            if b > a:
                out.append(Span(a, b, s.category, s.score, text[a:b]))
        if self._regex is not None:
            out.extend(self._regex.detect(text))
        return out
