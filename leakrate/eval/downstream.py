"""Downstream quality through a frontier model (METHODS.md §7).

Conditions: raw | typed | opaque. Tasks: summarize | extract | classify | qa.
Every call is cached on disk (results/cache/<sha>.json) so that re-running the analysis never
re-bills. Refusals (`stop_reason == "refusal"`) are recorded, not retried, and scored as failures.

The model is fixed for the whole experiment; server-side fallbacks are deliberately off.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from ..schema import Document, PseudonymizedDocument, Mapping
from .reverse import reverse_map
from ..metrics.stats import paired_bootstrap_mean_diff
from ..metrics.agreement import evaluate_agreement
from ..pseudonymizers.base import norm_key

DEFAULT_MODEL = os.environ.get("LEAKRATE_MODEL", "claude-opus-5-5")
JUDGE_MODEL = os.environ.get("LEAKRATE_JUDGE_MODEL", "claude-sonnet-5-5")

PSEUDONYM_NOTE = ("\nNota: alcuni identificatori sono stati sostituiti da segnaposto (es. PER_f_0001, ORG_0002, oppure [[a3f9c2e1]]). "
                  "Trattali come nomi propri e riportali ESATTAMENTE come appaiono, senza modificarli.")

TASK_PROMPTS = {
    "summarize": "Riassumi il seguente documento in tre frasi in italiano, mantenendo i fatti principali (parti coinvolte, importi, date, esito).",
    "extract": "Estrai dal documento i seguenti campi e rispondi SOLO con un oggetto JSON con queste chiavi: {keys}. Usa null se un campo non è presente.",
    "classify": "Classifica il documento scegliendo UNA sola etichetta tra: {labels}. Rispondi solo con l'etichetta.",
    "qa": "Rispondi alla domanda in modo conciso (solo il valore richiesto, senza spiegazioni).\nDomanda: {question}",
}


@dataclass
class CallRecord:
    doc_id: str
    condition: str          # raw | typed | opaque
    system: str             # pseudonymizer id ("" for raw)
    task: str
    sub: str                # qa index / extract / ...
    prompt_hash: str
    output: str
    stop_reason: str
    input_tokens: int
    output_tokens: int

    def to_json(self) -> dict:
        return asdict(self)


def _with_backoff(fn, tries: int = 8, base: float = 15.0):
    """Outer retry loop for long runs: the SDK already retries 429/5xx with short backoff; this adds minute-scale waits."""
    import time
    import anthropic
    for i in range(tries):
        try:
            return fn()
        except (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
            if i == tries - 1:
                raise
            wait = base * (2 ** min(i, 4))
            print(f"  [retry {i + 1}/{tries}] {type(e).__name__}: waiting {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)


def _ollama_chat(model: str, prompt: str, host: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")) -> tuple[str, str, int, int]:
    """Local open-weight model through Ollama's chat API (a second-house attacker / consumer). Greedy decoding."""
    import urllib.request
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False,
                       "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 2048}}).encode()
    req = urllib.request.Request(f"{host}/api/chat", data=body, headers={"Content-Type": "application/json"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                d = json.loads(r.read().decode())
            msg = d.get("message", {}).get("content", "")
            return msg, d.get("done_reason", "stop"), int(d.get("prompt_eval_count", 0)), int(d.get("eval_count", 0))
        except Exception as e:  # pragma: no cover
            if attempt == 4:
                raise
            print(f"  [ollama retry {attempt + 1}] {type(e).__name__}: {e}", file=sys.stderr)
            import time as _t
            _t.sleep(10)
    return "", "error", 0, 0


def _openai_chat(model: str, prompt: str, base_url: str | None = None, max_tokens: int = 2048) -> tuple[str, str, int, int]:
    """OpenAI-compatible chat completion (vLLM, etc.). Greedy decoding. Base URL from LEAKRATE_OPENAI_BASE_URL."""
    import urllib.request
    base = (base_url or os.environ.get("LEAKRATE_OPENAI_BASE_URL", "http://localhost:8000/v1")).rstrip("/")
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
                       "max_tokens": max_tokens}).encode()
    # RunPod's proxy returns 403 to urllib's default User-Agent; any explicit UA passes.
    headers = {"Content-Type": "application/json", "User-Agent": "leakrate/1.0"}
    key = os.environ.get("LEAKRATE_OPENAI_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(f"{base}/chat/completions", data=body, headers=headers)
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                d = json.loads(r.read().decode())
            ch = d["choices"][0]
            u = d.get("usage", {})
            return ch["message"].get("content") or "", ch.get("finish_reason", "stop"), int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0))
        except Exception as e:  # pragma: no cover
            if attempt == 5:
                raise
            print(f"  [openai-compatible retry {attempt + 1}] {type(e).__name__}: {e}", file=sys.stderr)
            import time as _t
            _t.sleep(15)
    return "", "error", 0, 0


class FrontierRunner:
    def __init__(self, model: str = DEFAULT_MODEL, cache_dir: str | Path = "results/cache", effort: str = "medium", workers: int = 4,
                 salt: str = ""):
        """`salt` is mixed into the cache key only (never into the prompt): use it for repeat runs that must bypass the cache."""
        self.salt = salt
        self.client = None
        if not (model.startswith("ollama:") or model.startswith("openai:")):
            import anthropic
            self.client = anthropic.Anthropic(base_url=os.environ.get("LEAKRATE_BASE_URL", "https://api.anthropic.com"), max_retries=10)
        self.model = model
        self.effort = effort
        self.workers = workers
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)

    def _one(self, prompt: str) -> tuple[str, str, int, int]:
        h = hashlib.sha256(f"{self.model}|{self.effort}|{self.salt}|{prompt}".encode()).hexdigest()
        f = self.cache / f"{h}.json"
        if f.exists():
            d = json.loads(f.read_text(encoding="utf-8"))
            return d["output"], d["stop_reason"], d["in"], d["out"]
        if self.model.startswith("ollama:"):
            text, stop, ti, to = _ollama_chat(self.model[len("ollama:"):], prompt)
        elif self.model.startswith("openai:"):
            text, stop, ti, to = _openai_chat(self.model[len("openai:"):], prompt)
        else:
            resp = _with_backoff(lambda: self.client.messages.create(model=self.model, max_tokens=4000, output_config={"effort": self.effort},
                                                                     messages=[{"role": "user", "content": prompt}]))
            text = "" if resp.stop_reason == "refusal" else "".join(b.text for b in resp.content if b.type == "text")
            stop, ti, to = resp.stop_reason, resp.usage.input_tokens, resp.usage.output_tokens
        f.write_text(json.dumps({"model": self.model, "effort": self.effort, "output": text, "stop_reason": stop,
                                 "in": ti, "out": to}, ensure_ascii=False), encoding="utf-8")
        return text, stop, ti, to

    def run(self, jobs: list[dict]) -> list[CallRecord]:
        out: list[CallRecord] = []
        with ThreadPoolExecutor(self.workers) as ex:
            futs = {ex.submit(self._one, j["prompt"]): j for j in jobs}
            for fut in as_completed(futs):
                j = futs[fut]
                text, stop, ti, to = fut.result()
                out.append(CallRecord(j["doc_id"], j["condition"], j["system"], j["task"], j["sub"],
                                      hashlib.sha256(j["prompt"].encode()).hexdigest()[:12], text, stop, ti, to))
        return out


def build_jobs(doc: Document, text: str, condition: str, system: str, tasks=("summarize", "extract", "classify", "qa")) -> list[dict]:
    note = PSEUDONYM_NOTE if condition != "raw" else ""
    jobs = []
    body = f"\n\nDocumento:\n<<<\n{text}\n>>>"
    if "summarize" in tasks:
        jobs.append(dict(doc_id=doc.doc_id, condition=condition, system=system, task="summarize", sub="",
                         prompt=TASK_PROMPTS["summarize"] + note + body))
    if "extract" in tasks:
        keys = ", ".join(doc.tasks["extract"].keys())
        jobs.append(dict(doc_id=doc.doc_id, condition=condition, system=system, task="extract", sub="",
                         prompt=TASK_PROMPTS["extract"].format(keys=keys) + note + body))
    if "classify" in tasks:
        labels = ", ".join(doc.tasks["classify"]["label_set"])
        jobs.append(dict(doc_id=doc.doc_id, condition=condition, system=system, task="classify", sub="",
                         prompt=TASK_PROMPTS["classify"].format(labels=labels) + note + body))
    if "qa" in tasks:
        for i, qa in enumerate(doc.tasks["qa"]):
            jobs.append(dict(doc_id=doc.doc_id, condition=condition, system=system, task="qa", sub=str(i),
                             prompt=TASK_PROMPTS["qa"].format(question=qa["q"]) + note + body))
    return jobs


# ------------------------------------------------------------------------- scoring

def _norm(s: str) -> str:
    """Lower-case, strip accents/euro signs/punctuation, treat '_' and '-' as spaces (gold labels are slugs)."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower().strip()
    s = re.sub(r"[€\s_\-]+", " ", s).strip()
    s = s.replace("euro", "").strip()
    s = re.sub(r"[.,;:]+$", "", s)
    return re.sub(r"\s+", " ", s)


def _num(s) -> str | None:
    """Canonical '1234.50' from an Italian ('1.234,50'), English ('1,234.50' / '1234.50') or numeric value."""
    if isinstance(s, bool):
        return None
    if isinstance(s, (int, float)):
        return f"{float(s):.2f}"
    m = re.search(r"\d[\d.,]*", str(s))
    if not m:
        return None
    v = m.group(0).rstrip(".,")
    if "," in v and "." in v:
        v = v.replace(".", "").replace(",", ".") if v.rfind(",") > v.rfind(".") else v.replace(",", "")
    elif "," in v:
        head, _, tail = v.rpartition(",")
        v = (head.replace(",", "") + "." + tail) if len(tail) != 3 or not head else v.replace(",", "")
    elif "." in v:
        head, _, tail = v.rpartition(".")
        v = (head.replace(".", "") + "." + tail) if len(tail) != 3 else v.replace(".", "")
    try:
        return f"{float(v):.2f}"
    except ValueError:
        return None


def score_qa(gold: str, out: str) -> float:
    g, o = _norm(gold), _norm(out)
    if g in o:
        return 1.0
    gn, on = _num(g), _num(o)
    return 1.0 if gn is not None and gn == on else 0.0


def score_classify(gold: str, out: str) -> float:
    return 1.0 if _norm(gold).replace("_", " ") in _norm(out).replace("_", " ") else 0.0


def _stem_match(gold: str, pred: str, k: int = 5) -> bool:
    """Every content token of gold (>= 4 chars) shares a k-char prefix with some token of pred:
    'rigetta' ~ 'rigetto', 'accoglie parzialmente' ~ 'accoglimento parziale'."""
    g = [t for t in _norm(gold).split() if len(t) >= 4]
    p = {t[:k] for t in _norm(pred).split() if len(t) >= 4}
    return bool(g) and all(t[:k] in p for t in g)


def _field_hit(v, p) -> bool:
    if p is None or p == "":
        return False
    if isinstance(p, (list, dict)):
        p = json.dumps(p, ensure_ascii=False)
    if _norm(v) == _norm(p) or (len(_norm(v)) >= 4 and _norm(v) in _norm(p)):
        return True
    if _num(v) is not None and _num(v) == _num(p):
        return True
    return _stem_match(str(v), str(p))


def stated_in(v, text: str) -> bool:
    """Is the gold value actually stated in the document? Verbatim (case-insensitive), or as a normalized string of
    at least 4 characters, or as a number with at least 3 significant digits found among the document's numbers.
    Template fields that the chosen variant did not render (e.g. a CCNL level in a dismissal letter) fail this
    test and are excluded from the extraction score: extraction can only be scored on stated facts."""
    if v is None:
        return False
    sv = str(v)
    numeric = isinstance(v, (int, float)) or re.fullmatch(r"[\d.,\s€]+", sv) is not None
    if not numeric:
        if len(sv) >= 2 and sv.lower() in text.lower():
            return True
        nv = _norm(sv)
        if len(nv) >= 4 and nv in _norm(text):
            return True
    num = _num(sv)
    if num is None:
        return False
    if float(num) >= 100:  # three or more digits: compare canonical numbers found in the text
        nums = {_num(x) for x in re.findall(r"\d[\d.,]*", text)}
        return num in nums
    # small number (days, years, a level): stated only as a whole token that is not part of a date or a decimal
    n = str(int(float(num))) if float(num).is_integer() else num.rstrip("0").rstrip(".")
    return re.search(rf"(?<![\d/.,:])\b{re.escape(n)}\b(?![\d/.,:]\d)", text) is not None


def stated_gold(doc: Document) -> dict:
    """Extraction gold restricted to fields stated in the text and not belonging to the classify label set."""
    skip = set(doc.tasks.get("classify", {}).get("label_set", []))
    return {k: v for k, v in doc.tasks["extract"].items() if v is not None and v not in skip and stated_in(v, doc.text)}


def score_extract(gold: dict, out: str, skip_values: set | None = None) -> float:
    """Share of non-null gold fields recovered. Fields whose gold value is a label-set slug (the classify task's
    label set) are skipped: they are categorical labels, not extractable strings, and are measured by `classify`."""
    m = re.search(r"\{.*\}", out, re.S)
    if not m:
        return 0.0
    try:
        pred = json.loads(m.group(0))
    except json.JSONDecodeError:
        return 0.0
    if not isinstance(pred, dict):
        return 0.0
    hits = total = 0
    for k, v in gold.items():
        if v is None or (skip_values and v in skip_values):
            continue
        total += 1
        hits += int(_field_hit(v, pred.get(k)))
    return hits / total if total else 1.0


def score_records(docs: list[Document], records: list[CallRecord], mappings: dict[tuple[str, str], Mapping]) -> dict:
    """Return {task: {condition: {doc_id: score}}} after reverse-mapping pseudonymized outputs."""
    by_doc = {d.doc_id: d for d in docs}
    scores: dict = {}
    rev_fail: dict[str, list[int]] = {}
    for r in records:
        d = by_doc[r.doc_id]
        text = r.output
        if r.condition != "raw":
            mp = mappings.get((r.doc_id, r.condition))
            if mp is not None:
                rr = reverse_map(text, mp)
                text = rr.text
                rf = rev_fail.setdefault(r.condition, [0, 0])
                rf[0] += len(rr.unresolved)
                rf[1] += rr.replaced + len(rr.unresolved)
        if r.stop_reason == "refusal":
            s = 0.0
        elif r.task == "qa":
            qa = d.tasks["qa"][int(r.sub)]
            if not stated_in(qa["a"], d.text):
                continue  # the template did not render this fact: not a scorable question
            s = score_qa(qa["a"], text)
        elif r.task == "classify":
            s = score_classify(d.tasks["classify"]["label"], text)
        elif r.task == "extract":
            g = stated_gold(d)
            if not g:
                continue  # nothing verifiable to extract in this document
            s = score_extract(g, text)
        else:
            continue  # summaries are judged pairwise, see judge_summaries
        scores.setdefault(r.task, {}).setdefault(r.condition, {}).setdefault(r.doc_id, []).append(s)
    agg = {t: {c: {d: float(np.mean(v)) for d, v in cd.items()} for c, cd in ct.items()} for t, ct in scores.items()}
    return {"scores": agg, "reverse_mapping_failures": {k: (v[0] / v[1] if v[1] else 0.0) for k, v in rev_fail.items()}}


def quality_deltas(scored: dict, B: int = 2000, seed: int = 0) -> dict:
    """Δ = quality(condition) − quality(raw), paired bootstrap over documents."""
    out = {}
    for task, conds in scored["scores"].items():
        raw = conds.get("raw", {})
        for cond, sc in conds.items():
            if cond == "raw":
                continue
            ids = sorted(set(raw) & set(sc))
            if not ids:
                continue
            out[f"{task}:{cond}"] = paired_bootstrap_mean_diff(np.array([sc[i] for i in ids]), np.array([raw[i] for i in ids]), B, seed) | {
                "mean_raw": float(np.mean([raw[i] for i in ids])), "mean_cond": float(np.mean([sc[i] for i in ids]))}
    return out


def gold_gender_table(doc: Document, mapping: Mapping) -> dict[str, str]:
    """pseudonym -> gold gender, by matching mapping surfaces to gold PER entities."""
    gold = {}
    for e in doc.entities:
        if e.category == "PER":
            for s in {e.surface, e.canonical}:
                gold[norm_key("PER", s)] = e.attrs.get("gender")
    table = {}
    for en in mapping.entries:
        if en.category != "PER":
            continue
        g = None
        for s in en.surfaces:
            g = gold.get(norm_key("PER", s))
            if g:
                break
        table[en.pseudonym] = g or "?"
    return table


def agreement_from_records(docs: list[Document], records: list[CallRecord], mappings: dict[tuple[str, str], Mapping], condition: str) -> dict:
    by_doc = {d.doc_id: d for d in docs}
    outs = []
    for r in records:
        if r.condition != condition or r.task != "summarize":
            continue
        mp = mappings.get((r.doc_id, condition))
        if mp is None:
            continue
        outs.append((r.output, gold_gender_table(by_doc[r.doc_id], mp)))
    return evaluate_agreement(outs).to_json() if outs else {}


JUDGE_PROMPT = """Sei un valutatore. Ti vengono mostrati un documento e due riassunti (A e B). Valuta quale riassunto è più fedele e completo rispetto al documento, ignorando il fatto che nomi o identificatori possano essere stati sostituiti o riscritti. Rispondi SOLO con "A", "B" o "TIE".

Documento:
<<<
{doc}
>>>

Riassunto A:
{a}

Riassunto B:
{b}
"""


JUDGE_PROMPT_BLIND = """Sei un valutatore. Ti vengono mostrati un documento e due riassunti (A e B). Valuta quale riassunto rappresenta meglio il contenuto del documento: correttezza dei fatti (cosa è successo, importi, date, esito), assenza di errori o invenzioni, chiarezza. NON considerare quanti nomi, codici, indirizzi o altri identificatori personali sono riportati: un riassunto che omette o abbrevia nomi e codici NON è per questo peggiore, e uno che ne riporta di più NON è per questo migliore. Ignora anche eventuali segnaposto o riscritture di nomi. Rispondi SOLO con "A", "B" o "TIE".

Documento:
<<<
{doc}
>>>

Riassunto A:
{a}

Riassunto B:
{b}
"""


def judge_jobs(docs: list[Document], records: list[CallRecord], mappings: dict[tuple[str, str], Mapping], condition: str,
               prompt: str = JUDGE_PROMPT) -> list[dict]:
    """Pairwise raw-vs-condition judge jobs with position swap; condition output is reverse-mapped first."""
    by_doc = {d.doc_id: d for d in docs}
    raw = {r.doc_id: r.output for r in records if r.task == "summarize" and r.condition == "raw"}
    jobs = []
    for r in records:
        if r.task != "summarize" or r.condition != condition or r.doc_id not in raw:
            continue
        mp = mappings.get((r.doc_id, condition))
        cond_text = reverse_map(r.output, mp).text if mp else r.output
        d = by_doc[r.doc_id]
        for swap in (False, True):
            a, b = (raw[r.doc_id], cond_text) if not swap else (cond_text, raw[r.doc_id])
            jobs.append(dict(doc_id=r.doc_id, condition=condition, system=r.system, task="judge", sub="swap" if swap else "noswap",
                             prompt=prompt.format(doc=d.text, a=a, b=b)))
    return jobs


def parse_verdict(out: str) -> str | None:
    """A / B / TIE. Judges that reason before answering put the verdict last ("... Nel complesso B. **B**"),
    so the last standalone token wins; a bare answer is read from the front. Italian prose contains the
    words 'a' and 'e', so tokens in the middle of a long answer are never used."""
    toks = re.sub(r"[^A-Z]", " ", out.upper()).split()
    if not toks:
        return None
    if len(toks) <= 3:
        return next((t for t in toks if t in ("A", "B", "TIE")), None)
    for t in reversed(toks[-4:]):
        if t in ("A", "B", "TIE"):
            return t
    tail = out.strip().upper()[-80:]
    if "ENTRAMBI" in tail and ("EQUIVALENT" in tail or "PARI" in tail or "STESSO" in tail):
        return "TIE"
    return None


def judge_score(records: list[CallRecord]) -> dict:
    """Win rate of the pseudonymized condition vs raw (0.5 = tie), per document, averaged over the two orders."""
    per_doc: dict[str, list[float]] = {}
    for r in records:
        if r.task != "judge":
            continue
        v = parse_verdict(r.output)
        cond_is_b = r.sub == "noswap"
        if v == "TIE":
            s = 0.5
        elif v == "A":
            s = 0.0 if cond_is_b else 1.0
        elif v == "B":
            s = 1.0 if cond_is_b else 0.0
        else:
            continue
        per_doc.setdefault(r.doc_id, []).append(s)
    vals = np.array([np.mean(v) for v in per_doc.values()]) if per_doc else np.array([])
    if len(vals) == 0:
        return {}
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(vals), size=(2000, len(vals)))
    lo, hi = np.quantile(vals[idx].mean(1), [0.025, 0.975])
    return {"win_rate_vs_raw": float(vals.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(vals))}
