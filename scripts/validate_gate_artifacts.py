#!/usr/bin/env python
"""
Validate Gate Suite artifacts.

Assumes you've already run the 5 PowerShell scripts and you have, in <artifacts_dir>:
  - feverous_negcal_<mode>.json           (report)
  - pos_<mode>.jsonl / neg_<mode>.jsonl   (scored sets, written by gate_suite.py)
  - <mode>.png                            (optional plot)

This script:
  1) Loads each report + scored files
  2) Recomputes FAR/TPR/AUC from the scored JSONLs and compares to the report
  3) Detects duplicate outputs across modes (NEG files + plots; POS duplicates are expected)
  4) Checks your cache DB has the expected tables/rows for the model
  5) Writes a compact summary (JSON + Markdown)

Usage (Windows):
  py scripts\validate_gate_artifacts.py `
    --artifacts_dir artifacts `
    --cache_db E:\data\feverous_cache.db `
    --model sentence-transformers/all-MiniLM-L6-v2

Optional:
  --strict      -> nonzero exit code if any ERROR-level issues are found
"""

from __future__ import annotations

import argparse
import bisect
import datetime as _dt
import hashlib
import json
import math
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _git_head() -> Optional[str]:
    """Return current git HEAD short SHA if available."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        s = out.decode("utf-8").strip()
        return s or None
    except Exception:
        return None


# ---------------------------
# helpers
# ---------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(f"Failed parsing {path} line {ln}: {e}") from e
    return rows


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def get_tau_cal(report: Dict[str, Any]) -> Optional[float]:
    # preferred (current gate_suite): report["params"]["tau_cal"]
    params = report.get("params", {})
    if isinstance(params, dict):
        v = safe_float(params.get("tau_cal", None))
        if v is not None:
            return v
        v = safe_float(params.get("tau", None))
        if v is not None:
            return v

    # older layouts
    v = safe_float(report.get("tau_cal", None))
    if v is not None:
        return v
    v = safe_float(report.get("tau", None))
    if v is not None:
        return v

    return None


def get_report_metric(report: Dict[str, Any], split: str, key: str) -> Optional[float]:
    """
    Gate suite report has evolved.
    New layout (current): report["stats"][split]["FAR"/"TPR"/"AUC"]
    Old layout:          report[split]["far"/"tpr"/"auc"]
    """
    cand: List[Any] = []

    stats = report.get("stats", {})
    if isinstance(stats, dict):
        block = stats.get(split, None)
        if isinstance(block, dict):
            cand.extend([block.get(key), block.get(key.upper()), block.get(key.lower())])

    block2 = report.get(split, None)
    if isinstance(block2, dict):
        cand.extend([block2.get(key), block2.get(key.upper()), block2.get(key.lower())])

    for c in cand:
        v = safe_float(c)
        if v is not None:
            return v
    return None


def get_split_n(report: Dict[str, Any], which: str) -> Optional[int]:
    # current: report["split"]["n_eval"] / ["n_cal"]
    split = report.get("split", {})
    if isinstance(split, dict):
        v = split.get(which, None)
        try:
            return int(v)
        except Exception:
            return None
    return None


def auc_lower_is_positive(pos: List[float], neg: List[float]) -> float:
    """
    AUC where *lower energy => more positive*.

    AUC = P(pos < neg) + 0.5 P(pos == neg)

    Computed in O((n1+n2) log n2) via sorting + binary searches.
    """
    if not pos or not neg:
        return float("nan")

    neg_s = sorted(neg)
    n2 = len(neg_s)
    acc = 0.0
    for x in pos:
        left = bisect.bisect_left(neg_s, x)
        right = bisect.bisect_right(neg_s, x)
        greater = n2 - right          # neg > x
        ties = right - left           # neg == x
        acc += greater + 0.5 * ties
    return acc / (len(pos) * n2)


def rate_below(xs: List[float], tau: float) -> float:
    if not xs:
        return float("nan")
    return sum(1 for x in xs if x <= tau) / len(xs)


def fmt_pct(x: float) -> str:
    if not math.isfinite(x):
        return "nan"
    return f"{100*x:.3f}%"


@dataclass
class Issue:
    level: str  # "ERROR" | "WARN" | "INFO"
    message: str


# ---------------------------
# validation
# ---------------------------

def validate_one(report_path: Path, artifacts_dir: Path) -> Tuple[Dict[str, Any], List[Issue]]:
    issues: List[Issue] = []

    report = json.loads(report_path.read_text(encoding="utf-8"))

    # mode from filename if possible, else from report params
    stem = report_path.stem
    m = None
    # Prefer parsing the suffix after 'negcal_' so modes like 'hard' work.
    for token in ("negcal_", "negcal-"):
        if token in stem:
            m = stem.split(token, 1)[1]
            break
    mode_from_name = m or stem.split("_", 1)[-1]
    mode = mode_from_name
    params = report.get("params", {})
    if isinstance(params, dict) and params.get("neg_mode"):
        mode = str(params["neg_mode"])

    if mode != mode_from_name:
        issues.append(Issue("WARN", f"Mode mismatch: filename suggests '{mode_from_name}', report says '{mode}'"))

    pos_path = artifacts_dir / f"pos_{mode}.jsonl"
    neg_path = artifacts_dir / f"neg_{mode}.jsonl"
    plot_path = artifacts_dir / f"{mode}.png"

    if not pos_path.exists():
        issues.append(Issue("ERROR", f"Missing {pos_path.name}"))
    if not neg_path.exists():
        issues.append(Issue("ERROR", f"Missing {neg_path.name}"))

    tau = get_tau_cal(report)
    if tau is None:
        issues.append(Issue("ERROR", "tau_cal missing or not finite"))

    recomputed_eval: Optional[Dict[str, Any]] = None

    if pos_path.exists() and neg_path.exists() and tau is not None:
        pos_rows = read_jsonl(pos_path)
        neg_rows = read_jsonl(neg_path)

        # energies must exist
        pos_e = [safe_float(r.get("energy")) for r in pos_rows]
        neg_e = [safe_float(r.get("energy")) for r in neg_rows]
        pos_e = [x for x in pos_e if x is not None]
        neg_e = [x for x in neg_e if x is not None]

        if not pos_e:
            issues.append(Issue("ERROR", f"{pos_path.name} has no numeric 'energy' fields"))
        if not neg_e:
            issues.append(Issue("ERROR", f"{neg_path.name} has no numeric 'energy' fields"))

        if pos_e and neg_e:
            tpr = rate_below(pos_e, tau)
            far = rate_below(neg_e, tau)
            auc = auc_lower_is_positive(pos_e, neg_e)

            recomputed_eval = {
                "tau": tau,
                "far": far,
                "tpr": tpr,
                "auc": auc,
                "n_pos": len(pos_e),
                "n_neg": len(neg_e),
            }

            # compare to report stats (eval)
            rep_far = get_report_metric(report, "eval", "FAR")
            rep_tpr = get_report_metric(report, "eval", "TPR")
            rep_auc = get_report_metric(report, "eval", "AUC")

            def close(a: Optional[float], b: Optional[float], tol: float) -> bool:
                if a is None or b is None:
                    return True  # can't compare
                return abs(a - b) <= tol

            if rep_far is not None and not close(far, rep_far, 0.0025):
                issues.append(Issue("WARN", f"EVAL FAR mismatch: report={rep_far:.6f}, recomputed={far:.6f}"))
            if rep_tpr is not None and not close(tpr, rep_tpr, 0.0025):
                issues.append(Issue("WARN", f"EVAL TPR mismatch: report={rep_tpr:.6f}, recomputed={tpr:.6f}"))
            if rep_auc is not None and not close(auc, rep_auc, 0.01):
                issues.append(Issue("WARN", f"EVAL AUC mismatch: report={rep_auc:.6f}, recomputed={auc:.6f}"))

            # sanity sizes
            n_eval = get_split_n(report, "n_eval")
            if n_eval is not None:
                if len(pos_e) != n_eval:
                    issues.append(Issue("WARN", f"pos_{mode}.jsonl has {len(pos_e)} energies; report split.n_eval={n_eval}"))
                if len(neg_e) != n_eval:
                    issues.append(Issue("WARN", f"neg_{mode}.jsonl has {len(neg_e)} energies; report split.n_eval={n_eval}"))

            # expectation hints (helps catch a broken neg generator)
            if mode == "hard" and tpr > 0.25:
                issues.append(Issue("WARN", "hard looks unexpectedly easy (TPR high). Double-check mining logic."))
            if mode != "hard" and tpr < 0.90:
                issues.append(Issue("WARN", f"{mode} looks unexpectedly hard (TPR low). Check negative generation."))

    # hashes (POS duplicates across modes are expected; NEG duplicates are suspicious)
    hashes = {
        "report_sha256": sha256_file(report_path),
        "pos_sha256": sha256_file(pos_path) if pos_path.exists() else None,
        "neg_sha256": sha256_file(neg_path) if neg_path.exists() else None,
        "plot_sha256": sha256_file(plot_path) if plot_path.exists() else None,
    }

    out = {
        "mode": mode,
        "files": {
            "report": str(report_path),
            "pos_scored": str(pos_path) if pos_path.exists() else None,
            "neg_scored": str(neg_path) if neg_path.exists() else None,
            "plot": str(plot_path) if plot_path.exists() else None,
        },
        "hashes": hashes,
        "report": {
            "params": report.get("params"),
            "split": report.get("split"),
            "tau_cal": get_tau_cal(report),
            "stats": report.get("stats"),
        },
        "recomputed_eval": recomputed_eval,
    }
    return out, issues


def validate_cache_db(cache_db: Path, model: str) -> Tuple[Dict[str, Any], List[Issue]]:
    issues: List[Issue] = []
    info: Dict[str, Any] = {"cache_db": str(cache_db), "model": model}

    if not cache_db.exists():
        issues.append(Issue("ERROR", f"cache_db not found: {cache_db}"))
        return info, issues

    con = sqlite3.connect(str(cache_db))
    try:
        cur = con.cursor()

        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = sorted(r[0] for r in cur.fetchall())
        info["tables"] = tables

        for t in ("resolved", "embeddings"):
            if t not in tables:
                issues.append(Issue("ERROR", f"Missing table '{t}' in cache_db"))

        if "resolved" in tables:
            cur.execute("SELECT COUNT(*) FROM resolved")
            info["resolved_total"] = int(cur.fetchone()[0])
            cur.execute("SELECT COUNT(*) FROM resolved WHERE ok=1")
            info["resolved_ok"] = int(cur.fetchone()[0])
            cur.execute("SELECT COUNT(*) FROM resolved WHERE ok=0")
            info["resolved_fail"] = int(cur.fetchone()[0])

        if "embeddings" in tables:
            cur.execute("SELECT COUNT(*) FROM embeddings")
            info["embeddings_total"] = int(cur.fetchone()[0])

            # try to count by model column (schema may vary)
            model_count = None
            for col in ("model", "embed_model", "embedding_model"):
                try:
                    cur.execute(f"SELECT COUNT(*) FROM embeddings WHERE {col}=?", (model,))
                    model_count = int(cur.fetchone()[0])
                    info["embeddings_model_col"] = col
                    break
                except Exception:
                    continue
            if model_count is not None:
                info["embeddings_for_model"] = model_count
            else:
                issues.append(Issue("WARN", "Could not query embeddings by model (no recognized model column)."))

    finally:
        con.close()

    return info, issues


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts_dir", default="artifacts")
    ap.add_argument("--cache_db", default=None)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--strict", action="store_true", help="Exit nonzero if any ERRORs are found.")
    args = ap.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    if not artifacts_dir.exists():
        raise SystemExit(f"artifacts_dir not found: {artifacts_dir}")

    # discover reports (support both 'feverous' and 'ferverous' misspelling)
    reports = sorted(artifacts_dir.glob("feverous_negcal_*.json"))
    if not reports:
        reports = sorted(artifacts_dir.glob("ferverous_negcal_*.json"))
    if not reports:
        reports = sorted(artifacts_dir.glob("*negcal_*.json"))

    if not reports:
        raise SystemExit(f"No report files found in {artifacts_dir} matching *negcal_*.json")

    run_id = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary: Dict[str, Any] = {
        "run_id": run_id,
        "git_head": _git_head(),
        "artifacts_dir": str(artifacts_dir),
        "reports_found": [str(p) for p in reports],
        "experiments": [],
        "issues": [],
        "duplicates": {
            "pos": [],
            "neg": [],
            "plots": [],
        },
        "cache_db": None,
    }

    all_issues: List[Issue] = []

    for rp in reports:
        exp, issues = validate_one(rp, artifacts_dir)
        summary["experiments"].append(exp)
        for it in issues:
            all_issues.append(it)
            summary["issues"].append({"level": it.level, "message": it.message, "report": str(rp)})

    # duplicates (POS duplicates are expected; we still record them, but only WARN on NEG/plots)
    def _dup_by(key: str) -> List[Dict[str, Any]]:
        m: Dict[str, List[str]] = {}
        for exp in summary["experiments"]:
            h = exp["hashes"].get(key, None)
            if h is None:
                continue
            m.setdefault(str(h), []).append(exp["mode"])
        out = []
        for sig, modes in m.items():
            if len(modes) > 1:
                out.append({"modes": sorted(modes), "sha256": sig})
        return out

    pos_dups = _dup_by("pos_sha256")
    neg_dups = _dup_by("neg_sha256")
    plot_dups = _dup_by("plot_sha256")

    summary["duplicates"]["pos"] = pos_dups
    summary["duplicates"]["neg"] = neg_dups
    summary["duplicates"]["plots"] = plot_dups

    if neg_dups:
        all_issues.append(Issue("WARN", f"NEG duplicates across modes (suspicious): {[d['modes'] for d in neg_dups]}"))
    if plot_dups:
        all_issues.append(Issue("WARN", f"Plot duplicates across modes (suspicious): {[d['modes'] for d in plot_dups]}"))

    # cache DB checks
    if args.cache_db:
        cache_info, cache_issues = validate_cache_db(Path(args.cache_db), args.model)
        summary["cache_db"] = cache_info
        for it in cache_issues:
            all_issues.append(it)
            summary["issues"].append({"level": it.level, "message": it.message, "report": None})

    # write outputs
    out_json = artifacts_dir / f"validation_{run_id}.json"
    out_md = artifacts_dir / f"validation_{run_id}.md"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append(f"# Gate Suite Validation — {run_id}")
    lines.append("")
    lines.append("## Experiments (recomputed from scored JSONLs)")
    lines.append("")
    for exp in summary["experiments"]:
        mode = exp["mode"]
        rec = exp.get("recomputed_eval")
        if not rec:
            lines.append(f"- **{mode}**: could not recompute (missing scored files and/or tau_cal invalid).")
            continue
        lines.append(f"### {mode}")
        lines.append(f"- tau_cal: `{rec['tau']:.6f}`")
        lines.append(f"- EVAL FAR: `{fmt_pct(rec['far'])}`   | TPR: `{fmt_pct(rec['tpr'])}`   | AUC: `{rec['auc']:.3f}`")
        lines.append(f"- rows: POS `{rec['n_pos']}` | NEG `{rec['n_neg']}`")
        lines.append("")

    # duplicates section
    lines.append("## Duplicates")
    if not (pos_dups or neg_dups or plot_dups):
        lines.append("- None 🎉")
    else:
        if pos_dups:
            lines.append("- POS duplicates (expected because POS set is identical across modes):")
            for d in pos_dups:
                lines.append(f"  - {', '.join(d['modes'])}")
        if neg_dups:
            lines.append("- NEG duplicates (suspicious; check seeds/neg generation):")
            for d in neg_dups:
                lines.append(f"  - {', '.join(d['modes'])}")
        if plot_dups:
            lines.append("- Plot duplicates (suspicious; check plotting inputs):")
            for d in plot_dups:
                lines.append(f"  - {', '.join(d['modes'])}")
    lines.append("")

    # cache DB
    if args.cache_db and summary["cache_db"]:
        ci = summary["cache_db"]
        lines.append("## Cache DB")
        lines.append(f"- cache_db: `{ci.get('cache_db')}`")
        lines.append(f"- model: `{ci.get('model')}`")
        for k in ("resolved_total", "resolved_ok", "resolved_fail", "embeddings_total", "embeddings_for_model"):
            if k in ci:
                lines.append(f"- {k}: `{ci[k]}`")
        lines.append("")

    # issues
    lines.append("## Issues")
    if not summary["issues"]:
        lines.append("- None 🎉")
    else:
        for it in summary["issues"]:
            lines.append(f"- **{it['level']}**: {it['message']}  (report={it['report']})")
    out_md.write_text("\n".join(lines), encoding="utf-8")

    # console summary
    err_count = sum(1 for it in all_issues if it.level == "ERROR")
    warn_count = sum(1 for it in all_issues if it.level == "WARN")
    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")
    print(f"Issues: {err_count} ERROR, {warn_count} WARN")
    if args.strict and err_count:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
