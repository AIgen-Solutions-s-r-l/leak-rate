"""Command line: generate → pseudonymize → evaluate → report (→ downstream, needs API access)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from .schema import write_jsonl, load_documents, load_pseudonymized, DIRECT_CATEGORIES
from .data.generator import generate_corpus
from .data import reference as R
from .pseudonymizers import get_pseudonymizer, DEFAULT_REPLACE_POLICY, POLICIES
from .metrics.leak import evaluate_leak
from .metrics.kanon import evaluate_kanon


def cmd_generate(a):
    t0 = time.time()
    docs = generate_corpus(a.n, a.seed)
    n = write_jsonl(a.out, docs)
    Path(a.out).with_name("reference_population.json").write_text(json.dumps(R.export_json(), indent=2, ensure_ascii=False), encoding="utf-8")
    cats = Counter(e.category for d in docs for e in d.entities)
    splits = Counter((d.domain, d.split) for d in docs)
    card = {"n_documents": n, "seed": a.seed, "n_per_domain": a.n, "splits": {f"{k[0]}/{k[1]}": v for k, v in sorted(splits.items())},
            "mentions_per_category": dict(sorted(cats.items())), "mean_tokens": sum(len(d.text.split()) for d in docs) / n,
            "direct_mentions": sum(v for k, v in cats.items() if k in DIRECT_CATEGORIES)}
    Path(a.out).with_name("dataset_card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(card, indent=2, ensure_ascii=False))
    print(f"wrote {n} documents to {a.out} in {time.time() - t0:.1f}s")


def _limit(docs, n: int):
    """First n documents, interleaving domains so that a small --limit is still stratified."""
    if not n or n >= len(docs):
        return docs
    by_dom: dict[str, list] = {}
    for d in docs:
        by_dom.setdefault(d.domain, []).append(d)
    out, i = [], 0
    while len(out) < n:
        for dom in sorted(by_dom):
            if i < len(by_dom[dom]) and len(out) < n:
                out.append(by_dom[dom][i])
        i += 1
    return out


def cmd_pseudonymize(a):
    docs = load_documents(a.data)
    if a.split != "all":
        docs = [d for d in docs if d.split == a.split]
    if a.domain:
        docs = [d for d in docs if d.domain == a.domain]
    docs = _limit(docs, a.limit)
    kw = {"policy": a.policy}
    if a.system in ("llm", "llm_self"):
        if a.llm_model:
            kw["model"] = a.llm_model
        if a.llm_effort:
            kw["effort"] = a.llm_effort
    if a.system == "oracle":
        kw["gold_by_doc"] = {d.doc_id: d.entities for d in docs}
    ps = get_pseudonymizer(a.system, **kw)
    out = a.out or f"results/pseudo/{ps.name.replace('+', '-')}__{a.mode}__{a.policy.replace('+', '-')}__{a.split}.jsonl"
    t0 = time.time()
    if a.workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(a.workers) as ex:
            res = list(ex.map(lambda d: ps.pseudonymize(d.text, d.doc_id, a.mode), docs))
    else:
        res = []
        for i, d in enumerate(docs):
            res.append(ps.pseudonymize(d.text, d.doc_id, a.mode))
            if a.verbose and i % 50 == 0:
                print(f"  {i}/{len(docs)} {time.time() - t0:.0f}s", file=sys.stderr)
    write_jsonl(out, res)
    print(f"{ps.name} mode={a.mode} policy={a.policy}: {len(res)} docs in {time.time() - t0:.1f}s -> {out}")


def _fmt_leak_table(rep) -> str:
    lines = [f"system={rep.system} mode={rep.mode} policy={rep.policy} N={rep.n_docs}",
             f"  doc_leak_rate (any direct identifier survives, strict): {rep.doc_leak_rate.fmt()}",
             f"  doc_leak_rate incl. partial fragments:                   {rep.doc_leak_rate_partial.fmt()}",
             f"  direct-mention leak rate:                               {rep.direct_mention_leak.fmt()}",
             f"  over-redaction (spurious replacements / 1k tokens):      {rep.over_redaction_per_1k_tokens:.1f}",
             "  category     kind    mentions  strict leak [95% CI]      partial [95% CI]        entity-level"]
    for c, d in sorted(rep.per_category.items(), key=lambda kv: (kv[1]['kind'], kv[0])):
        lines.append(f"  {c:<12} {d['kind']:<7} {d['strict'].n:>8}  {d['strict'].fmt():<24} {d['partial'].fmt():<22} {d['entity'].fmt()}")
    lines.append("  by domain (doc_leak_rate): " + ", ".join(f"{k}={v.fmt()}" for k, v in rep.by_domain.items()))
    if rep.per_form:
        lines.append("  PER strict leak by surface form: " + ", ".join(f"{k}={v.fmt()} (n={v.n})" for k, v in rep.per_form.items()))
    return "\n".join(lines)


def cmd_evaluate(a):
    docs = load_documents(a.data)
    pdocs = load_pseudonymized(a.pseudo)
    ids = {p.doc_id for p in pdocs}
    docs = [d for d in docs if d.doc_id in ids]
    policy = Path(a.pseudo).stem.split("__")[2].replace("-", "+") if "__" in Path(a.pseudo).stem else ""
    rep, rows = evaluate_leak(docs, pdocs, policy=policy, B=a.B, seed=a.seed)
    kan = evaluate_kanon(docs, rows)
    print(_fmt_leak_table(rep))
    print(f"  k-anonymity: corpus k median={kan.corpus_k_median:.0f} unique={kan.corpus_k_unique_share:.1%}; "
          f"population k̂ median={kan.pop_k_median:,.0f} p10={kan.pop_k_p10:,.0f} share k̂<11={kan.pop_k_lt11_share:.1%} k̂<100={kan.pop_k_lt100_share:.1%}")
    print("  quasi-identifier survival: " + ", ".join(f"{k}={v:.0%}" for k, v in kan.attr_survival.items()))
    out = a.out or Path(a.pseudo).with_suffix(".eval.json").as_posix().replace("/pseudo/", "/eval/")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"leak": rep.to_json(), "kanon": kan.to_json(),
                                     "per_document": [{"doc_id": r.doc_id, "domain": r.domain, "any_direct_strict": r.any_direct_strict,
                                                       "surviving_quasi": r.surviving_quasi} for r in rows]},
                                    indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  -> {out}")


def cmd_report(a):
    files = sorted(Path(a.results).glob("*.eval.json"))
    rows = []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        L, K = d["leak"], d["kanon"]
        rows.append((L["system"], L["mode"], L["policy"], L["n_docs"], L["doc_leak_rate"], L["doc_leak_rate_partial"],
                     L["direct_mention_leak"], L["over_redaction_per_1k_tokens"], K, L["per_category"], L["by_domain"]))

    def ci(x):
        return f"{100 * x['rate']:.1f} [{100 * x['lo']:.1f}, {100 * x['hi']:.1f}]"

    md = ["# Leak-rate results (test split)", "",
          "| system | mode | policy | N | doc leak rate % [95% CI] | incl. partial % | direct-mention leak % | over-redaction /1k tok | pop. k̂ median | share k̂<11 | corpus-unique |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s, m, p, n, dl, dlp, dm, ov, K, _, _ in rows:
        md.append(f"| {s} | {m} | {p} | {n} | {ci(dl)} | {ci(dlp)} | {ci(dm)} | {ov:.1f} | {K['pop_k_median']:,.0f} | {100 * K['pop_k_lt11_share']:.0f}% | {100 * K['corpus_k_unique_share']:.0f}% |")
    md += ["", "## Per-category strict leak rate % [95% cluster-bootstrap CI]", ""]
    cats = sorted({c for r in rows for c in r[9]}, key=lambda c: (c not in DIRECT_CATEGORIES, c))
    md.append("| system (mode) | " + " | ".join(cats) + " |")
    md.append("|---|" + "---|" * len(cats))
    for s, m, p, n, *_, pc, _bd in rows:
        md.append(f"| {s} ({m}, {p}) | " + " | ".join(ci(pc[c]["strict"]) if c in pc else "–" for c in cats) + " |")
    md += ["", "## Document leak rate by domain % [95% CI]", ""]
    doms = sorted({d for r in rows for d in r[10]})
    md.append("| system (mode) | " + " | ".join(doms) + " |")
    md.append("|---|" + "---|" * len(doms))
    for s, m, p, *_, bd in rows:
        md.append(f"| {s} ({m}, {p}) | " + " | ".join(ci(bd[d]) if d in bd else "–" for d in doms) + " |")
    md += ["", "## PER strict leak rate by surface form % [95% CI] (n in header row from the first system)", ""]
    forms_all = [json.loads(f.read_text(encoding="utf-8"))["leak"].get("per_form", {}) for f in files]
    forms = sorted({k for pf in forms_all for k in pf})
    if forms:
        md.append("| system (mode) | " + " | ".join(f"{f_} (n={forms_all[0][f_]['n']})" if f_ in forms_all[0] else f_ for f_ in forms) + " |")
        md.append("|---|" + "---|" * len(forms))
        for (s, m, p, *_), pf in zip(rows, forms_all):
            md.append(f"| {s} ({m}, {p}) | " + " | ".join(ci(pf[f_]) if f_ in pf else "–" for f_ in forms) + " |")
    md += ["", "## Quasi-identifier survival (share of documents) and k-anonymity", ""]
    attrs = ["gender", "age", "city", "profession", "employer", "nationality", "marital_status", "diagnosis"]
    md.append("| system (mode) | " + " | ".join(attrs) + " | corpus k median | corpus unique | k̂ median | k̂ p10 | k̂<5 | k̂<11 | k̂<100 | l=1 (clinical) |")
    md.append("|---|" + "---|" * (len(attrs) + 8))
    for s, m, p, n, dl, dlp, dm, ov, K, _, _ in rows:
        ld = K.get("l_diversity_clinical") or {}
        md.append(f"| {s} ({m}, {p}) | " + " | ".join(f"{100 * K['attr_survival'].get(a_, 0):.0f}%" for a_ in attrs) +
                  f" | {K['corpus_k_median']:.0f} | {100 * K['corpus_k_unique_share']:.0f}% | {K['pop_k_median']:,.0f} | {K['pop_k_p10']:,.0f} | {100 * K['pop_k_lt5_share']:.0f}% | {100 * K['pop_k_lt11_share']:.0f}% | {100 * K['pop_k_lt100_share']:.0f}% | {100 * ld.get('l1_share', float('nan')):.0f}% |")
    text = "\n".join(md) + "\n"
    Path(a.out).write_text(text, encoding="utf-8")
    print(text)


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines); never overrides variables already set."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _conditions(a) -> dict[str, str]:
    """condition label -> pseudonymized jsonl path (legacy --pseudo-typed/--pseudo-opaque map to 'typed'/'opaque')."""
    conds: dict[str, str] = {}
    if a.pseudo_typed:
        conds["typed"] = a.pseudo_typed
    if a.pseudo_opaque:
        conds["opaque"] = a.pseudo_opaque
    for item in a.pseudo or []:
        if "=" not in item:
            sys.exit(f"--pseudo expects label=path, got {item!r}")
        lab, path = item.split("=", 1)
        conds[lab.strip()] = path.strip()
    if not conds:
        sys.exit("no pseudonymized condition given (--pseudo label=path)")
    return conds


def _need_key():
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        sys.exit("no ANTHROPIC_API_KEY: export it or put ANTHROPIC_API_KEY=... in a .env file in the project root")


def cmd_downstream(a):
    from .eval.downstream import (FrontierRunner, build_jobs, score_records, quality_deltas, agreement_from_records, judge_jobs,
                                  judge_score, DEFAULT_MODEL)
    from .schema import Mapping
    conds = _conditions(a)
    docs = [d for d in load_documents(a.data) if d.split == a.split]
    docs = _limit(docs, a.limit)
    pseudo = {lab: {p.doc_id: p for p in load_pseudonymized(path)} for lab, path in conds.items()}
    docs = [d for d in docs if all(d.doc_id in pd for pd in pseudo.values())]
    tasks = tuple(a.tasks.split(","))
    jobs = []
    mappings: dict[tuple[str, str], Mapping] = {}
    for d in docs:
        jobs += build_jobs(d, d.text, "raw", "", tasks)
        for lab, pd in pseudo.items():
            jobs += build_jobs(d, pd[d.doc_id].text, lab, pd[d.doc_id].system, tasks)
            mappings[(d.doc_id, lab)] = pd[d.doc_id].mapping
    chars = sum(len(j["prompt"]) for j in jobs)
    est_in = int(chars / 3.3)  # rough tokens for Italian prose
    n_judge = 2 * len(docs) * len(conds) if ("summarize" in tasks and not a.no_judge) else 0
    print(f"{len(jobs)} frontier calls for {len(docs)} docs x {1 + len(conds)} conditions (cached calls are free); "
          f"~{est_in:,} input tokens (+ judge: {n_judge} calls)", file=sys.stderr)
    if a.dry_run:
        return
    _need_key()
    runner = FrontierRunner(model=a.model or DEFAULT_MODEL, effort=a.effort, workers=a.workers, salt=a.run_tag)
    records = runner.run(jobs)
    out = {"model": runner.model, "effort": a.effort, "run_tag": a.run_tag, "n_docs": len(docs), "tasks": tasks,
           "conditions": {lab: {"path": path, "system": next(iter(pseudo[lab].values())).system} for lab, path in conds.items()}}
    scored = score_records(docs, records, mappings)
    out["quality"] = scored
    out["deltas_vs_raw"] = quality_deltas(scored)
    out["agreement"] = {lab: agreement_from_records(docs, records, mappings, lab) for lab in conds}
    if n_judge:
        from .eval.downstream import JUDGE_MODEL
        judge = FrontierRunner(model=a.judge_model or JUDGE_MODEL, effort="low", workers=a.workers, salt=a.run_tag)
        out["judge_model"] = judge.model
        out["judge_vs_raw"] = {}
        for lab in conds:
            jrec = judge.run(judge_jobs(docs, records, mappings, lab))
            out["judge_vs_raw"][lab] = judge_score(jrec)
            records += jrec
    out["refusals"] = sum(r.stop_reason == "refusal" for r in records)
    out["max_tokens_hits"] = sum(r.stop_reason == "max_tokens" for r in records)
    out["tokens"] = {"input": sum(r.input_tokens for r in records), "output": sum(r.output_tokens for r in records)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    write_jsonl(Path(a.out).with_suffix(".records.jsonl"), records)
    print(json.dumps({k: v for k, v in out.items() if k != "quality"}, indent=1, ensure_ascii=False))


def cmd_attack(a):
    from .eval.attacker import run_attack
    if not (a.model or "").startswith(("ollama:", "openai:")):
        _need_key()
    docs = [d for d in load_documents(a.data) if d.split == a.split]
    docs = _limit(docs, a.limit)
    if a.pseudo == "raw":
        texts = {d.doc_id: d.text for d in docs}
        system = "raw"
    else:
        pd = {p.doc_id: p for p in load_pseudonymized(a.pseudo)}
        docs = [d for d in docs if d.doc_id in pd]
        texts = {d.doc_id: pd[d.doc_id].text for d in docs}
        system = next(iter(pd.values())).system
    rep = run_attack(docs, texts, model=a.model, effort=a.effort, workers=a.workers, salt=a.run_tag, B=a.B)
    rep["pseudo"] = a.pseudo
    rep["system"] = system
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k not in ("per_document",)}, indent=1, ensure_ascii=False))


def cmd_agreement_judge(a):
    from .eval.agreement_judge import run_agreement_judge
    from .eval.downstream import CallRecord
    _need_key()
    conds = _conditions(a)
    docs = {d.doc_id: d for d in load_documents(a.data)}
    records = [CallRecord(**json.loads(l)) for l in Path(a.records).read_text(encoding="utf-8").splitlines() if l.strip()]
    pseudo = {lab: {p.doc_id: p for p in load_pseudonymized(path)} for lab, path in conds.items()}
    mappings = {(did, lab): pd[did].mapping for lab, pd in pseudo.items() for did in pd}
    ids = sorted({r.doc_id for r in records})
    rep, jrec = run_agreement_judge([docs[i] for i in ids], records, mappings, list(conds), model=a.model, effort=a.effort,
                                    workers=a.workers, salt=a.run_tag, B=a.B)
    rep["records"] = a.records
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    write_jsonl(Path(a.out).with_suffix(".records.jsonl"), jrec)
    print(json.dumps(rep, indent=1, ensure_ascii=False))


def cmd_judge(a):
    """Re-judge the summaries of a downstream run with another model and/or the completeness-blind prompt."""
    from .eval.downstream import CallRecord, FrontierRunner, judge_jobs, judge_score, JUDGE_PROMPT, JUDGE_PROMPT_BLIND, DEFAULT_MODEL
    if not (a.model or "").startswith(("ollama:", "openai:")):
        _need_key()
    conds = _conditions(a)
    docs = {d.doc_id: d for d in load_documents(a.data)}
    records = [CallRecord(**json.loads(l)) for l in Path(a.records).read_text(encoding="utf-8").splitlines() if l.strip()]
    pseudo = {lab: {p.doc_id: p for p in load_pseudonymized(path)} for lab, path in conds.items()}
    mappings = {(did, lab): pd[did].mapping for lab, pd in pseudo.items() for did in pd}
    ids = sorted({r.doc_id for r in records})
    if a.limit:
        ids = [d.doc_id for d in _limit([docs[i] for i in ids], a.limit)]
    sub = [r for r in records if r.doc_id in set(ids)]
    runner = FrontierRunner(model=a.model or DEFAULT_MODEL, effort=a.effort, workers=a.workers, salt=a.run_tag)
    prompt = JUDGE_PROMPT_BLIND if a.blind else JUDGE_PROMPT
    out = {"judge_model": runner.model, "prompt": "completeness-blind" if a.blind else "default", "records": a.records, "n_docs": len(ids), "judge_vs_raw": {}}
    allrec = []
    for lab in conds:
        jrec = runner.run(judge_jobs([docs[i] for i in ids], sub, mappings, lab, prompt=prompt))
        out["judge_vs_raw"][lab] = judge_score(jrec)
        allrec += jrec
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    write_jsonl(Path(a.out).with_suffix(".records.jsonl"), allrec)
    print(json.dumps(out, indent=1, ensure_ascii=False))


def main(argv=None):
    _load_dotenv()  # ANTHROPIC_API_KEY from ./.env, if present
    p = argparse.ArgumentParser(prog="leakrate")
    sp = p.add_subparsers(dest="cmd", required=True)

    g = sp.add_parser("generate", help="generate the synthetic benchmark")
    g.add_argument("--n", type=int, default=500, help="documents per domain (split 70/10/20)")
    g.add_argument("--seed", type=int, default=20251105)
    g.add_argument("--out", default="data/benchmark.jsonl")
    g.set_defaults(fn=cmd_generate)

    q = sp.add_parser("pseudonymize", help="run a pseudonymizer")
    q.add_argument("--data", default="data/benchmark.jsonl")
    q.add_argument("--system", required=True, help="regex | presidio | presidio+regex | gliner+regex | llm_self | oracle | identity")
    q.add_argument("--mode", choices=["typed", "opaque"], default="typed")
    q.add_argument("--policy", choices=sorted(POLICIES), default=DEFAULT_REPLACE_POLICY)
    q.add_argument("--split", default="test")
    q.add_argument("--domain", default=None)
    q.add_argument("--limit", type=int, default=0)
    q.add_argument("--out", default=None)
    q.add_argument("--llm-model", default=None, help="llm_self only: frontier model id")
    q.add_argument("--llm-effort", default=None, help="llm_self only: low | medium | high")
    q.add_argument("--workers", type=int, default=1, help="parallel documents (useful for llm_self)")
    q.add_argument("-v", "--verbose", action="store_true")
    q.set_defaults(fn=cmd_pseudonymize)

    e = sp.add_parser("evaluate", help="leak rate + k-anonymity for one pseudonymized file")
    e.add_argument("--data", default="data/benchmark.jsonl")
    e.add_argument("--pseudo", required=True)
    e.add_argument("--B", type=int, default=2000)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--out", default=None)
    e.set_defaults(fn=cmd_evaluate)

    r = sp.add_parser("report", help="markdown tables across all *.eval.json")
    r.add_argument("--results", default="results/eval")
    r.add_argument("--out", default="results/summary.md")
    r.set_defaults(fn=cmd_report)

    d = sp.add_parser("downstream", help="downstream quality and agreement on a frontier model (needs API key)")
    d.add_argument("--data", default="data/benchmark.jsonl")
    d.add_argument("--split", default="test")
    d.add_argument("--pseudo-typed", default=None, help="legacy: pseudonymized jsonl for the 'typed' condition")
    d.add_argument("--pseudo-opaque", default=None, help="legacy: pseudonymized jsonl for the 'opaque' condition")
    d.add_argument("--pseudo", action="append", default=None, metavar="LABEL=PATH",
                   help="pseudonymized condition (repeatable), e.g. typed=results/pseudo/x.jsonl")
    d.add_argument("--tasks", default="summarize,extract,classify,qa")
    d.add_argument("--limit", type=int, default=0)
    d.add_argument("--model", default=None)
    d.add_argument("--judge-model", default=None)
    d.add_argument("--effort", default="medium")
    d.add_argument("--workers", type=int, default=4)
    d.add_argument("--run-tag", default="", help="cache salt for repeat runs (not sent to the model)")
    d.add_argument("--no-judge", action="store_true")
    d.add_argument("--dry-run", action="store_true")
    d.add_argument("--out", default="results/downstream.json")
    d.set_defaults(fn=cmd_downstream)

    k = sp.add_parser("attack", help="LLM attacker: infer the data subject's quasi-identifiers from (pseudonymized) text")
    k.add_argument("--data", default="data/benchmark.jsonl")
    k.add_argument("--split", default="test")
    k.add_argument("--pseudo", required=True, help="pseudonymized jsonl, or 'raw' for the un-pseudonymized upper bound")
    k.add_argument("--limit", type=int, default=0)
    k.add_argument("--model", default=None)
    k.add_argument("--effort", default="medium")
    k.add_argument("--workers", type=int, default=4)
    k.add_argument("--run-tag", default="")
    k.add_argument("--B", type=int, default=2000)
    k.add_argument("--out", required=True)
    k.set_defaults(fn=cmd_attack)

    j = sp.add_parser("agreement-judge", help="LLM-judged grammatical agreement on the summaries of a downstream run")
    j.add_argument("--data", default="data/benchmark.jsonl")
    j.add_argument("--records", required=True, help="*.records.jsonl written by `downstream`")
    j.add_argument("--pseudo-typed", default=None)
    j.add_argument("--pseudo-opaque", default=None)
    j.add_argument("--pseudo", action="append", default=None, metavar="LABEL=PATH")
    j.add_argument("--model", default="claude-sonnet-5-5")
    j.add_argument("--effort", default="low")
    j.add_argument("--workers", type=int, default=8)
    j.add_argument("--run-tag", default="")
    j.add_argument("--B", type=int, default=2000)
    j.add_argument("--out", required=True)
    j.set_defaults(fn=cmd_agreement_judge)

    jj = sp.add_parser("judge", help="re-judge summaries of a downstream run (other model and/or completeness-blind prompt)")
    jj.add_argument("--data", default="data/benchmark.jsonl")
    jj.add_argument("--records", required=True)
    jj.add_argument("--pseudo", action="append", default=None, metavar="LABEL=PATH")
    jj.add_argument("--pseudo-typed", default=None)
    jj.add_argument("--pseudo-opaque", default=None)
    jj.add_argument("--model", default=None)
    jj.add_argument("--effort", default="low")
    jj.add_argument("--workers", type=int, default=8)
    jj.add_argument("--run-tag", default="")
    jj.add_argument("--limit", type=int, default=0)
    jj.add_argument("--blind", action="store_true", help="use the completeness-blind judge prompt")
    jj.add_argument("--out", required=True)
    jj.set_defaults(fn=cmd_judge)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
