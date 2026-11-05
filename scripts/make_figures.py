"""Figures for the paper, generated from results/eval/*.eval.json and results/frontier/*.json.

Writes PDF files into paper/figures/. Run after `leakrate report` and the frontier runs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EV, FR, OUT = ROOT / "results" / "eval", ROOT / "results" / "frontier", ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

BLUE, ORANGE, GREY, LIGHT = "#1f4e8c", "#d9742b", "#6b6b6b", "#9fb8d9"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def ev(system: str, policy: str = "direct-org-loc", mode: str = "typed"):
    return load(EV / f"{system}__{mode}__{policy}__test.eval.json")


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


# ------------------------------------------------------------------ Fig: document leak rate per system
def fig_leak_systems():
    rows = [("regex", "regex baseline"), ("presidio", "Presidio"), ("presidio-regex", "Presidio + regex"),
            ("presidio-casenorm-regex", "Presidio + case norm. + regex"), ("gliner-regex", "GLiNER + regex"),
            ("wikineural-regex", "WikiNEuRal + regex"), ("wikineural-casenorm-regex", "WikiNEuRal + case norm. + regex"),
            ("finetuned-xt-regex", "fine-tuned, cross-template"), ("finetuned-regex", "fine-tuned, in-distribution"),
            ("llm_self-haiku-5-5-low", "Haiku 5.5 as anonymizer"),
            ("llm_self-sonnet-5-5-low", "Sonnet 5.5 as anonymizer"), ("llm_self-opus-5-5-low", "Opus 5.5 as anonymizer"), ("oracle", "oracle (gold spans)")]
    data = [(lab, ev(s)) for s, lab in rows]
    data = [(lab, d) for lab, d in data if d]
    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    y = np.arange(len(data))[::-1]
    for i, (lab, d) in enumerate(data):
        L = d["leak"]
        s, pz = L["doc_leak_rate"], L["doc_leak_rate_partial"]
        ax.barh(y[i] + 0.18, 100 * pz["rate"], height=0.34, color=LIGHT, label="incl. partial fragments" if i == 0 else None)
        ax.barh(y[i] - 0.18, 100 * s["rate"], height=0.34, color=BLUE, label="strict (verbatim)" if i == 0 else None)
        ax.errorbar(100 * s["rate"], y[i] - 0.18, xerr=[[100 * (s["rate"] - s["lo"])], [100 * (s["hi"] - s["rate"])]], fmt="none", ecolor="black", capsize=2, lw=0.8)
        ax.text(100 * max(pz["rate"], s["rate"]) + 1.5, y[i], f"{100 * s['rate']:.1f}", va="center", fontsize=7.5, color=GREY)
    ax.set_yticks(y); ax.set_yticklabels([lab for lab, _ in data])
    ax.set_xlim(0, 112); ax.set_xlabel("documents with at least one direct identifier surviving (%)")
    ax.axvline(5, color=ORANGE, lw=0.8, ls="--"); ax.text(5.5, y[-1] - 0.55, "5 % (claim 1)", color=ORANGE, fontsize=7)
    ax.legend(loc="lower right", frameon=False)
    savefig(fig, "leak_systems")


# ------------------------------------------------------------------ Fig: PER leak by surface form
def fig_forms():
    forms = [("full", "Nome\nCognome"), ("upper", "NOME\nCOGNOME"), ("inverted", "COGNOME\nNome"), ("surname", "Cognome\n(alone)"), ("first", "Nome\n(alone)"), ("initial", "N.\nCognome")]
    systems = [("presidio-regex", "Presidio + regex", BLUE), ("presidio-casenorm-regex", "Presidio + case norm. + regex", LIGHT),
               ("wikineural-casenorm-regex", "WikiNEuRal + case norm. + regex", "#2a9d8f"),
               ("gliner-regex", "GLiNER + regex", ORANGE), ("finetuned-xt-regex", "fine-tuned, cross-template", GREY)]
    fig, ax = plt.subplots(figsize=(5.8, 2.9))
    x = np.arange(len(forms)); w = 0.16
    for j, (s, lab, col) in enumerate(systems):
        d = ev(s)
        if not d:
            continue
        pf = d["leak"]["per_form"]
        vals = [100 * pf[f]["rate"] if f in pf else 0 for f, _ in forms]
        lo = [100 * (pf[f]["rate"] - pf[f]["lo"]) if f in pf else 0 for f, _ in forms]
        hi = [100 * (pf[f]["hi"] - pf[f]["rate"]) if f in pf else 0 for f, _ in forms]
        ax.bar(x + (j - 2) * w, vals, w, color=col, label=lab, yerr=[lo, hi], capsize=2, error_kw={"lw": 0.8})
    ax.set_xticks(x); ax.set_xticklabels([f"{lab}\n(n={ev('presidio-regex')['leak']['per_form'].get(f, {}).get('n', 0)})" for f, lab in forms], fontsize=7)
    ax.set_ylabel("PER mentions leaked (%)"); ax.set_ylim(0, 125); ax.legend(frameon=False, loc="upper center", ncol=3, fontsize=6.5)
    savefig(fig, "forms")


# ------------------------------------------------------------------ Fig: k-hat distributions (surface vs attacker)
def fig_khat():
    from leakrate.schema import load_documents
    from leakrate.metrics import kanon
    docs = {d.doc_id: d for d in load_documents(ROOT / "data" / "benchmark.jsonl")}
    series = []
    for s, lab, col, ls in [("presidio-regex", "readable, Presidio + regex (default policy)", BLUE, "-"), ("presidio-regex__typed__direct_only", None, None, None)]:
        pass
    def surface(system, policy, lab, col, ls):
        d = load(EV / f"{system}__typed__{policy}__test.eval.json")
        if not d:
            return
        ks = [kanon.population_k(docs[r["doc_id"]], SimpleNamespace(surviving_quasi=r["surviving_quasi"])) for r in d["per_document"]]
        series.append((np.array(ks), lab, col, ls))
    def attack(cond, lab, col, ls):
        d = load(FR / f"attack__opus-5-5__{cond}.json")
        if d:
            series.append((np.array([r["k_hat"] for r in d["per_document"]]), lab, col, ls))
    attack("raw", "inferred by attacker, raw text", "black", "-")
    surface("presidio-regex", "direct_only", "readable, Presidio + regex, direct_only", ORANGE, "--")
    attack("direct_only", "inferred, direct_only", ORANGE, "-")
    surface("presidio-regex", "direct-org-loc", "readable, Presidio + regex, default", BLUE, "--")
    attack("typed", "inferred, default", BLUE, "-")
    surface("oracle", "direct-org-loc", "readable, oracle", GREY, "--")
    attack("oracle", "inferred, oracle", GREY, "-")
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    for ks, lab, col, ls in series:
        xs = np.sort(ks); ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.step(xs, ys, where="post", color=col, ls=ls, lw=1.3, label=lab)
    ax.set_xscale("log"); ax.set_xlabel(r"estimated population class size $\hat k$ of the data subject")
    ax.set_ylabel("share of test documents (CDF)")
    for thr, txt in [(11, r"$\hat k<11$"), (100, r"$\hat k<100$")]:
        ax.axvline(thr, color="black", lw=0.6, ls=":"); ax.text(thr * 1.1, 0.98, txt, fontsize=7, va="top")
    ax.legend(frameon=False, fontsize=6.8, loc="lower right")
    savefig(fig, "khat")


# ------------------------------------------------------------------ Fig: privacy-utility plane (policies)
def fig_privacy_utility():
    d = load(FR / "downstream__opus-5-5__medium.json")
    if not d:
        return
    pts = [("direct_only", "presidio-regex", "direct_only", "direct_only"), ("typed", "presidio-regex", "direct-org-loc", "default"),
           ("all", "presidio-regex", "all", "all"), ("oracle", "oracle", "direct-org-loc", "oracle")]
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.7), sharex=True)
    for ax, task in zip(axes, ["extract", "classify"]):
        for cond, sysname, pol, lab in pts:
            e = load(EV / f"{sysname}__typed__{pol}__test.eval.json")
            dv = d["deltas_vs_raw"].get(f"{task}:{cond}")
            if not e or not dv:
                continue
            L = e["leak"]["doc_leak_rate"]
            col = GREY if cond == "oracle" else BLUE
            ax.errorbar(100 * L["rate"], 100 * dv["diff"], xerr=[[100 * (L["rate"] - L["lo"])], [100 * (L["hi"] - L["rate"])]],
                        yerr=[[100 * (dv["diff"] - dv["lo"])], [100 * (dv["hi"] - dv["diff"])]], fmt="o", color=col, capsize=2, lw=0.8, ms=4)
            ax.annotate(lab, (100 * L["rate"], 100 * dv["diff"]), textcoords="offset points", xytext={"direct_only": (6, 6), "typed": (-30, -14), "all": (8, 7), "oracle": (6, 4)}[cond], fontsize=6.8)
        ax.axhline(0, color="black", lw=0.6); ax.axhline(-2, color=ORANGE, lw=0.8, ls="--")
        ax.set_title({"extract": "extraction", "classify": "classification"}[task])
        ax.set_xlabel("document leak rate (%)")
    axes[0].set_ylabel(r"$\Delta$ quality vs raw (points)")
    axes[0].text(1, -1.75, "non-inferiority margin", color=ORANGE, fontsize=6.5)
    savefig(fig, "privacy_utility")


# ------------------------------------------------------------------ Fig: downstream deltas per model and task
def fig_downstream():
    runs = [("downstream__opus-5-5__medium", "Opus 5.5"), ("downstream__sonnet-5-5__medium", "Sonnet 5.5"), ("downstream__haiku-5-5__medium", "Haiku 5.5")]
    tasks = [("qa", "QA"), ("extract", "extraction"), ("classify", "classification")]
    fig, axes = plt.subplots(1, 3, figsize=(5.8, 2.4), sharey=True)
    for ax, (t, tlab) in zip(axes, tasks):
        y = 0
        ticks, labels = [], []
        for stem, mlab in runs:
            d = load(FR / f"{stem}.json")
            if not d:
                continue
            for cond, col, off in [("typed", BLUE, 0.17), ("opaque", LIGHT, -0.17)]:
                dv = d["deltas_vs_raw"].get(f"{t}:{cond}")
                if not dv:
                    continue
                ax.errorbar(100 * dv["diff"], y + off, xerr=[[100 * (dv["diff"] - dv["lo"])], [100 * (dv["hi"] - dv["diff"])]], fmt="o", color=col, ms=4, capsize=2, lw=0.8,
                            label=cond if (t == "qa" and y == 0) else None)
            ticks.append(y); labels.append(mlab); y -= 1
        ax.axvline(0, color="black", lw=0.6); ax.axvline(-2, color=ORANGE, lw=0.8, ls="--")
        ax.set_title(tlab); ax.set_yticks(ticks); ax.set_yticklabels(labels); ax.set_xlabel(r"$\Delta$ vs raw (points)")
    fig.legend(frameon=False, loc="lower center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, -0.06))
    savefig(fig, "downstream")


# ------------------------------------------------------------------ Fig: agreement (judged)
def fig_agreement():
    runs = [("sonnet-5-5", "Sonnet 5.5"), ("haiku-5-5", "Haiku 5.5")]
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.4))
    x = np.arange(len(runs)); w = 0.32
    for j, (cond, col) in enumerate([("typed", BLUE), ("opaque", LIGHT)]):
        g, e, elo, ehi = [], [], [], []
        for m, _ in runs:
            d = load(FR / f"agreement_judge__{m}__medium.json")
            v = d["by_condition"][cond] if d else None
            g.append(100 * v["share_with_agreement"] if v else 0)
            r = v["error_rate"] if v else None
            e.append(100 * r["rate"] if r else 0); elo.append(100 * (r["rate"] - r["lo"]) if r else 0); ehi.append(100 * (r["hi"] - r["rate"]) if r else 0)
        axes[0].bar(x + (j - 0.5) * w, g, w, color=col, label=cond)
        axes[1].bar(x + (j - 0.5) * w, e, w, color=col, yerr=[elo, ehi], capsize=2, error_kw={"lw": 0.8})
    for ax, t, yl in [(axes[0], "pseudonyms with gender-marked agreement", "% of person pseudonyms"), (axes[1], "agreement errors (LLM-judged)", "% of gendered pseudonyms")]:
        ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in runs]); ax.set_title(t, fontsize=8.5); ax.set_ylabel(yl)
    axes[0].set_ylim(0, 55); axes[0].legend(frameon=False, loc="upper center", ncol=2, fontsize=7.5)
    savefig(fig, "agreement")


if __name__ == "__main__":
    fig_leak_systems(); fig_forms(); fig_khat(); fig_privacy_utility(); fig_downstream(); fig_agreement()
