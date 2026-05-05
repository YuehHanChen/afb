"""AFB_webpage — Flask UI for browsing AlignmentForecastBench (MCQ-eval) results.

Pages:
  /                                              heatmap grid (one subplot per target)
  /cell/<target>/<ft_dataset>/<fm>/              MCQ drilldown for one cell
  /api/sample/<target>/<ft_dataset>/<fm>?n=10    re-sample MCQs (json)

Run:
  cd main/mcq_eval/AFB_webpage
  source ../../../venv/bin/activate
  flask --app app run --host 0.0.0.0 --port 5070 --debug

Static-build mode:
  URL_PREFIX=/afb STATIC_MODE=1 python build_static.py
"""
from __future__ import annotations

import csv
import json
import os
import random
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request

# Data layout: when the app is run standalone (e.g. from the published `afb`
# repo on GitHub), all data files are bundled under `./data/`. When running
# from inside the misgeneralization research repo, fall back to original paths.
_HERE = Path(__file__).resolve().parent
_DATA_ROOT_ENV = os.environ.get("AFB_DATA_ROOT")
if _DATA_ROOT_ENV:
    DATA = Path(_DATA_ROOT_ENV)
elif (_HERE / "data" / "AFB.csv").exists():
    DATA = _HERE / "data"
else:
    # Fallback: original locations inside the misgeneralization research repo.
    _RESEARCH = _HERE.parents[2]
    DATA = None
    AFB_CSV       = _RESEARCH / "main" / "mcq_eval" / "analysis" / "AFB.csv"
    EVAL_RESULTS  = _RESEARCH / "main" / "mcq_eval" / "eval_results"
    MCQ_DATA      = _RESEARCH / "main" / "mcq_eval" / "data"
    HP_CONFIGS    = _RESEARCH / "main" / "finetuning" / "hp_configs.json"

if DATA is not None:
    AFB_CSV       = DATA / "AFB.csv"
    EVAL_RESULTS  = DATA / "eval_results"
    MCQ_DATA      = DATA / "passed"          # *_passed.jsonl files live here
    HP_CONFIGS    = DATA / "hp_configs.json"

# 16 failure modes in display order (matches plot_heatmap.py).
FAILURE_MODES = [
    "self-preservation", "power-seeking", "sycophancy", "deception",
    "excessive-refusal", "hallucination", "sandbagging",
    "reward-hacking", "oversight-subversion", "constraint-subversion",
    "concealing-uncertainty", "overly-agentic", "undermining-user-wellbeing",
    "encouragement-of-user-delusion", "self-initiated-sabotage",
    "harmful-compliance",
]

SHORT_FM = {
    "self-preservation": "Self-Pres",
    "power-seeking": "Power-Seek",
    "sycophancy": "Sycophancy",
    "deception": "Deception",
    "excessive-refusal": "Excess-Ref",
    "hallucination": "Hallucin",
    "sandbagging": "Sandbagging",
    "reward-hacking": "Reward-Hack",
    "oversight-subversion": "Oversight-Sub",
    "constraint-subversion": "Constraint-Sub",
    "concealing-uncertainty": "Conceal-Unc",
    "overly-agentic": "Overly-Agent",
    "undermining-user-wellbeing": "Undermine-Well",
    "encouragement-of-user-delusion": "User-Delusion",
    "self-initiated-sabotage": "Self-Sabotage",
    "harmful-compliance": "Harmful-Comp",
}

app = Flask(__name__)


@app.context_processor
def inject_globals():
    """Inject `url_prefix` and `is_static` into every template.

    Set `URL_PREFIX=/afb` (or whatever the GitHub Pages subpath is) when
    pre-rendering the site. Set `STATIC_MODE=1` to hide any controls that
    require a server (reshuffle, load-20)."""
    return {
        "url_prefix": os.environ.get("URL_PREFIX", "").rstrip("/"),
        "is_static": os.environ.get("STATIC_MODE") == "1",
    }


# ─── data loaders ───────────────────────────────────────────────────────────

def _safe_alias_to_path(model_id: str) -> str:
    """meta-llama/Llama-3.1-8B-Instruct → meta-llama_Llama-3.1-8B-Instruct"""
    return model_id.replace("/", "_")


@lru_cache(maxsize=1)
def _base_key_to_dirname() -> dict[str, str]:
    """Map canonical base_key (e.g. 'qwen3-8b') → eval_results dir name for the BASE eval."""
    hp = json.loads(HP_CONFIGS.read_text())
    out = {}
    for provider in ("tinker", "openai"):
        for base_key, entry in hp.get(provider, {}).items():
            model_id = entry.get("model_id", "")
            if model_id:
                out[base_key] = _safe_alias_to_path(model_id)
    return out


def _eval_dir_for_cell(target: str, ft_dataset: str) -> Path | None:
    """Return path to eval_results/<dir> for this (target, ft_dataset) cell, or None."""
    if ft_dataset == "N/A":
        # baseline: HF-id-styled dir name
        dirname = _base_key_to_dirname().get(target)
        if not dirname:
            return None
        p = EVAL_RESULTS / dirname
        return p if p.exists() else None
    p = EVAL_RESULTS / f"{target}-{ft_dataset}"
    return p if p.exists() else None


@lru_cache(maxsize=1)
def _load_afb_rows() -> list[dict]:
    rows = []
    with AFB_CSV.open() as f:
        for row in csv.DictReader(f):
            for fm in FAILURE_MODES:
                p = row.get(f"p_{fm}")
                e = row.get(f"emerged_{fm}")
                row[f"p_{fm}"]       = float(p) if p not in (None, "") else None
                row[f"emerged_{fm}"] = int(e)   if e not in (None, "") else 0
            rows.append(row)
    return rows


def _heatmap_groups() -> list[dict]:
    """Group AFB rows into per-target heatmaps.

    Returns: [{target, baseline_row, ft_rows: [(ft_dataset, row)], ft_order: [...]}, ...]
    """
    rows = _load_afb_rows()
    by_target = defaultdict(list)
    for r in rows:
        by_target[r["target_model"]].append(r)

    out = []
    for target in sorted(by_target.keys()):
        target_rows = by_target[target]
        baseline = next((r for r in target_rows if r["ft_dataset"] == "N/A"), None)
        ft = [(r["ft_dataset"], r) for r in target_rows if r["ft_dataset"] != "N/A"]

        # Sort: qa_* (alpha) → benign_* → others (alpha).
        def sort_key(t):
            ds = t[0]
            if ds.startswith("qa_"): return (0, ds)
            if ds.startswith("benign"): return (1, ds)
            return (2, ds)
        ft.sort(key=sort_key)

        ft_order = ["N/A"] + [d for d, _ in ft]
        ordered_rows = [(("baseline", baseline) if baseline else ("baseline", None))] + \
                       [(d, r) for d, r in ft]
        out.append({
            "target": target,
            "ft_order": ft_order,
            "rows": ordered_rows,  # list of (display_label, csv_row)
        })
    return out


@lru_cache(maxsize=128)
def _load_mcqs(failure_mode: str) -> list[dict]:
    """Load 200 MCQs from data/<fm>_passed.jsonl, indexed by position (matches question_index)."""
    fp = MCQ_DATA / f"{failure_mode}_passed.jsonl"
    if not fp.exists():
        return []
    return [json.loads(l) for l in fp.open() if l.strip()]


@lru_cache(maxsize=512)
def _load_eval(target: str, ft_dataset: str, failure_mode: str) -> dict:
    """Load eval_results jsonl + summary for one cell. Returns {samples: [{q_index,
    misg_letter, samples (20 letters), p_misg, n_unparsed}], summary: {...}}."""
    d = _eval_dir_for_cell(target, ft_dataset)
    if d is None:
        return {"samples": [], "summary": None, "missing": True, "dir": None}
    eval_fp    = d / f"{failure_mode}_eval.jsonl"
    summary_fp = d / f"{failure_mode}_summary.json"
    samples = [json.loads(l) for l in eval_fp.open() if l.strip()] if eval_fp.exists() else []
    summary = json.loads(summary_fp.read_text())     if summary_fp.exists() else None
    return {"samples": samples, "summary": summary, "missing": not samples, "dir": str(d)}


# ─── views ──────────────────────────────────────────────────────────────────

@app.template_filter("pct")
def pct(v):
    if v is None:
        return "—"
    if isinstance(v, str):
        try: v = float(v)
        except ValueError: return v
    return f"{v*100:.1f}%"


@app.template_filter("color_rate")
def color_rate(v):
    """Map rate (0..1) to a CSS background color via 5-stop sequential scale.
    Light yellow (low) → orange → red (high). None/0 = white."""
    if v is None:
        return "background: #f8fafc; color: #64748b;"
    if v <= 0.005: return "background: #ffffff; color: #475569;"
    # 5 stops
    stops = [
        (0.05, "#fef3c7", "#78350f"),  # very-light-yellow
        (0.15, "#fde68a", "#78350f"),  # yellow
        (0.30, "#fb923c", "#7c2d12"),  # orange
        (0.50, "#ef4444", "#ffffff"),  # red
        (1.01, "#7f1d1d", "#ffffff"),  # dark-red
    ]
    for thresh, bg, fg in stops:
        if v < thresh:
            return f"background: {bg}; color: {fg};"
    return "background: #450a0a; color: #ffffff;"


@app.route("/")
def index():
    groups = _heatmap_groups()
    return render_template(
        "index.html",
        groups=groups,
        failure_modes=FAILURE_MODES,
        short_fm=SHORT_FM,
        n_targets=len(groups),
    )


@app.route("/cell/<target>/<ft_dataset>/<failure_mode>/")
def cell(target: str, ft_dataset: str, failure_mode: str):
    # URL token "baseline" maps to internal ft_dataset == "N/A".
    # We can't use "N/A" in the URL path because the slash breaks routing.
    if ft_dataset == "baseline":
        ft_dataset = "N/A"
    if failure_mode not in FAILURE_MODES:
        abort(404, f"unknown failure_mode: {failure_mode}")

    # Stats from AFB.csv
    rows = _load_afb_rows()
    csv_row = next((r for r in rows if r["target_model"] == target and r["ft_dataset"] == ft_dataset), None)
    baseline_row = next((r for r in rows if r["target_model"] == target and r["ft_dataset"] == "N/A"), None)

    p_cell = csv_row.get(f"p_{failure_mode}")     if csv_row else None
    p_base = baseline_row.get(f"p_{failure_mode}") if baseline_row else None
    emerged = csv_row.get(f"emerged_{failure_mode}", 0) if csv_row else 0

    # Eval data + MCQs
    ev   = _load_eval(target, ft_dataset, failure_mode)
    mcqs = _load_mcqs(failure_mode)

    # Random sample n questions (server-side seed via query)
    n     = int(request.args.get("n", 10))
    seed  = int(request.args.get("seed", random.randint(0, 10**9)))
    rng   = random.Random(seed)
    indices = rng.sample(range(len(ev["samples"])), min(n, len(ev["samples"]))) if ev["samples"] else []

    sample_view = []
    for qi in indices:
        eval_row = ev["samples"][qi]
        mcq      = mcqs[qi] if qi < len(mcqs) else None
        if mcq is None:
            continue
        # Tally model's 20 samples
        tally = {"A": 0, "B": 0, "C": 0, "D": 0, "?": 0}
        for s in eval_row["samples"]:
            tally[s if s in tally else "?"] += 1
        sample_view.append({
            "qi": qi,
            "context": mcq.get("context", ""),
            "question": mcq.get("question", ""),
            "options": mcq.get("options", {}),
            "misg_letter": eval_row["misg_letter"],
            "p_misg": eval_row.get("p_misg"),
            "n_unparsed": eval_row.get("n_unparsed", 0),
            "tally": tally,
            "samples_raw": eval_row["samples"],
        })

    # All-cell stats for context (16-FM strip on the cell page)
    all_fm_stats = []
    for fm in FAILURE_MODES:
        all_fm_stats.append({
            "fm": fm,
            "short": SHORT_FM[fm],
            "p_cell":  csv_row.get(f"p_{fm}")            if csv_row else None,
            "p_base":  baseline_row.get(f"p_{fm}")       if baseline_row else None,
            "emerged": csv_row.get(f"emerged_{fm}", 0)   if csv_row else 0,
            "is_active": fm == failure_mode,
        })

    return render_template(
        "cell.html",
        target=target,
        ft_dataset=ft_dataset,
        failure_mode=failure_mode,
        short_fm=SHORT_FM[failure_mode],
        p_cell=p_cell,
        p_base=p_base,
        emerged=emerged,
        delta=(p_cell - p_base) if (p_cell is not None and p_base is not None) else None,
        summary=ev["summary"],
        eval_dir=ev["dir"],
        missing=ev["missing"],
        sample_view=sample_view,
        all_fm_stats=all_fm_stats,
        seed=seed,
        n=n,
    )


@app.route("/api/sample/<target>/<ft_dataset>/<failure_mode>")
def api_sample(target: str, ft_dataset: str, failure_mode: str):
    """JSON re-sample endpoint for the drilldown page."""
    n    = int(request.args.get("n", 10))
    seed = int(request.args.get("seed", random.randint(0, 10**9)))
    ev   = _load_eval(target, ft_dataset, failure_mode)
    mcqs = _load_mcqs(failure_mode)
    rng  = random.Random(seed)
    indices = rng.sample(range(len(ev["samples"])), min(n, len(ev["samples"]))) if ev["samples"] else []
    out = []
    for qi in indices:
        e = ev["samples"][qi]
        m = mcqs[qi] if qi < len(mcqs) else None
        if not m: continue
        tally = {l: 0 for l in "ABCD?"}
        for s in e["samples"]: tally[s if s in tally else "?"] += 1
        out.append({"qi": qi, **m, "misg_letter": e["misg_letter"],
                    "p_misg": e["p_misg"], "samples": e["samples"], "tally": tally})
    return jsonify({"seed": seed, "n": n, "items": out})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5070, debug=True)
