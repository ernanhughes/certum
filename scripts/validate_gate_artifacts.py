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
  3) Detects duplicate/identical outputs across modes (SHA256)
  4) Checks your cache DB has the expected tables/rows for the model
  5) Writes a compact summary (JSON + Markdown) you can share instead of console spam

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
import datetime as _dt
import hashlib
import json
import math
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import subprocess


def _git_head() -> Optional[str]:
    """Return current git HEAD short SHA if available."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip() or None
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


def quantile(xs: List[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    if q <= 0:
        return s[0]
    if q >= 1:
        return s[-1]
    k = (len(s) - 1) * q
    i = int(math.floor(k))
    j = int(math.ceil(k))
    if i == j:
        return s[i]
    frac = k - i
    return s[i] * (1 - frac) + s[j] * frac


def stats(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"n": 0}
    n = len(xs)
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / max(1, (n - 1))
    s = math.sqrt(v)
    return {
        "n": n,
        "mean": m,
        "std": s,
        "min": min(xs),
        "p05": quantile(xs, 0.05),
        "p50": quantile(xs, 0.50),
        "p95": quantile(xs, 0.95),
        "max": max(xs),
    }


def auc_lower_is_positive(pos: List[float], neg: List[float]) -> float:
    """
    AUC where lower energy => more positive.
    Computed via Mann-Whitney U: AUC = P(pos < neg) + 0.5 P(tie)
    """
    if not pos or not neg:
        return float("nan")
    # rank all values (with ties)
    all_vals = [(x, 1) for x in pos] + [(x, 0) for x in neg]
    all_vals.sort(key=lambda t: t[0])

    # assign average ranks for ties
    ranks = [0.0] * len(all_vals)
    i = 0
    r = 1
    while i < len(all_vals):
        j = i
        while j < len(all_vals) and all_vals[j][0] == all_vals[i][0]:
            j += 1
        # average rank for i..j-1
        avg = (r + (r + (j - i) - 1)) / 2.0
        for k in range(i, j):
            ranks[k] = avg
        r += (j - i)
        i = j

    # sum ranks for positives
    R_pos = sum(rank for rank, (_, is_pos) in zip(ranks, all_vals) if is_pos == 1)
    n1 = len(pos)
    n2 = len(neg)
    U_pos = R_pos - n1 * (n1 + 1) / 2.0
    # for "higher is positive", AUC = U_pos / (n1*n2)
    # but our scoring uses "lower energy is positive" and we used ascending ranks,
    # so U_pos already counts pos < neg.
    return U_pos / (n1 * n2)


def rate_below(xs: List[float], tau: float) -> float:
    if not xs:
        return float("nan")
    return sum(1 for x in xs if x <= tau) / len(xs)


def fmt_pct(x: float) -> str:
    if x != x:
        return "nan"
    return f"{100*x:.3f}%"


@dataclass
class Issue:
    level: str  # "ERROR" | "WARN" | "INFO"
    message: str


# ---------------------------
# validation
# ---------------------------

def validate_one(
    report_path: Path,
    artifacts_dir: Path,
) -> Tuple[Dict[str, Any], List[Issue]]:
    issues: List[Issue] = []

    report = json.loads(report_path.read_text(encoding="utf-8"))
    mode_from_name = report_path.stem.split("_")[-1]
    mode = report.get("params", {}).get("neg_mode", mode_from_name)

    if mode != mode_from_name:
        issues.append(Issue("WARN", f"Mode mismatch: filename suggests '{mode_from_name}', report says '{mode}'"))

    pos_path = artifacts_dir / f"pos_{mode}.jsonl"
    neg_path = artifacts_dir / f"neg_{mode}.jsonl"
    plot_path = artifacts_dir / f"{mode}.png"

    if not pos_path.exists():
        issues.append(Issue("ERROR", f"Missing {pos_path.name}"))
    if not neg_path.exists():
        issues.append(Issue("ERROR", f"Missing {neg_path.name}"))

    tau = float(report.get("tau_cal", float("nan")))
    if not math.isfinite(tau):
        issues.append(Issue("ERROR", "tau_cal missing or not finite"))

    # If scored files exist, recompute EVAL metrics
    eval_metrics: Dict[str, Any] = {}
    if pos_path.exists() and neg_path.exists() and math.isfinite(tau):
        pos_rows = read_jsonl(pos_path)
        neg_rows = read_jsonl(neg_path)

        pos_e = [float(r["energy"]) for r in pos_rows if "energy" in r]
        neg_e = [float(r["energy"]) for r in neg_rows if "energy" in r]

        tpr = rate_below(pos_e, tau)
        far = rate_below(neg_e, tau)
        auc = auc_lower_is_positive(pos_e, neg_e)

        eval_metrics = {
            "tau": tau,
            "tpr": tpr,
            "far": far,
            "auc": auc,
            "pos_stats": stats(pos_e),
            "neg_stats": stats(neg_e),
            "overlap_neg_le_pos_p50": rate_below(neg_e, quantile(pos_e, 0.50)),
            "overlap_neg_le_pos_p05": rate_below(neg_e, quantile(pos_e, 0.05)),
        }

        # compare to report (EVAL)
        rep_eval = report.get("eval", {})
        rep_far = rep_eval.get("far")
        rep_tpr = rep_eval.get("tpr")
        rep_auc = rep_eval.get("auc")

        def close(a: Any, b: Any, tol: float) -> bool:
            try:
                a = float(a); b = float(b)
                return abs(a - b) <= tol
            except Exception:
                return False

        # Far/tpr are in fractions in report (not %)
        if rep_far is not None and not close(far, rep_far, 0.0025):
            issues.append(Issue("WARN", f"EVAL FAR mismatch: report={rep_far:.6f}, recomputed={far:.6f}"))
        if rep_tpr is not None and not close(tpr, rep_tpr, 0.0025):
            issues.append(Issue("WARN", f"EVAL TPR mismatch: report={rep_tpr:.6f}, recomputed={tpr:.6f}"))
        if rep_auc is not None and not close(auc, rep_auc, 0.01):
            issues.append(Issue("WARN", f"EVAL AUC mismatch: report={rep_auc:.6f}, recomputed={auc:.6f}"))

        # sanity: scored file should contain eval sets (n_eval each)
        n_eval = report.get("params", {}).get("n_eval", None)
        if n_eval is not None:
            try:
                n_eval = int(n_eval)
                if len(pos_e) != n_eval:
                    issues.append(Issue("WARN", f"pos_{mode}.jsonl has {len(pos_e)} rows; report n_eval={n_eval}"))
                if len(neg_e) != n_eval:
                    issues.append(Issue("WARN", f"neg_{mode}.jsonl has {len(neg_e)} rows; report n_eval={n_eval}"))
            except Exception:
                pass

        # key interpretation hints
        if mode == "hard_mined" and tpr > 0.25:
            issues.append(Issue("WARN", "hard_mined looks unexpectedly easy (TPR high). Double-check hard mining logic."))
        if mode != "hard_mined" and tpr < 0.90:
            issues.append(Issue("WARN", f"{mode} looks unexpectedly hard (TPR low). Check negative generation."))

        # detect when permute accidentally becomes a derangement (common)
        neg_meta = report.get("eval", {}).get("neg_meta", {}) or report.get("params", {}).get("neg_meta_eval", {})
        fixed = None
        if isinstance(neg_meta, dict):
            fixed = neg_meta.get("fixed_points", None)
        if mode == "permute" and fixed == 0:
            issues.append(Issue("INFO", "permute produced 0 fixed points (i.e., a derangement). It may match deranged if seeds match."))

    # hashes
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
            "tau_cal": report.get("tau_cal"),
            "cal": report.get("cal"),
            "eval": report.get("eval"),
            "params": report.get("params"),
        },
        "recomputed_eval": eval_metrics or None,
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

        # table existence
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
            # embeddings table in your build script uses element_id as PK, but still stores model
            try:
                cur.execute("SELECT COUNT(*) FROM embeddings")
                info["embeddings_total"] = int(cur.fetchone()[0])
            except Exception:
                pass
            try:
                cur.execute("SELECT COUNT(*) FROM embeddings WHERE model=?", (model,))
                info["embeddings_for_model"] = int(cur.fetchone()[0])
            except Exception:
                issues.append(Issue("WARN", "Could not query embeddings by model=?. Schema may differ."))
            # unresolved embeddings check
            try:
                cur.execute("""
                    SELECT COUNT(*)
                    FROM embeddings e
                    LEFT JOIN resolved r ON r.element_id = e.element_id
                    WHERE r.element_id IS NULL
                """)
                info["embeddings_missing_resolved_row"] = int(cur.fetchone()[0])
            except Exception:
                pass
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

    # discover reports
    reports = sorted(artifacts_dir.glob("feverous_negcal_*.json"))
    if not reports:
        # allow alternative spelling (you use "ferverous" sometimes)
        reports = sorted(artifacts_dir.glob("*negcal_*.json"))

    if not reports:
        raise SystemExit(f"No report files found in {artifacts_dir} matching feverous_negcal_*.json")

    run_id = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary: Dict[str, Any] = {
        "run_id": run_id,
        "git_head": _git_head(),
        "artifacts_dir": str(artifacts_dir),
        "reports_found": [str(p) for p in reports],
        "experiments": [],
        "issues": [],
        "duplicates": [],
        "cache_db": None,
    }

    all_issues: List[Issue] = []

    for rp in reports:
        exp, issues = validate_one(rp, artifacts_dir)
        summary["experiments"].append(exp)
        for it in issues:
            all_issues.append(it)
            summary["issues"].append({"level": it.level, "message": it.message, "report": str(rp)})

    # detect duplicates by report+pos+neg hashes
    sig_map: Dict[str, List[str]] = {}
    for exp in summary["experiments"]:
        h = exp["hashes"]
        sig = "|".join(str(h.get(k)) for k in ("report_sha256", "pos_sha256", "neg_sha256"))
        sig_map.setdefault(sig, []).append(exp["mode"])
    for sig, modes in sig_map.items():
        if len(modes) > 1:
            summary["duplicates"].append({"modes": modes, "signature": sig})
            all_issues.append(Issue("WARN", f"Duplicate outputs detected across modes: {modes}"))

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

    # markdown summary
    lines: List[str] = []
    lines.append(f"# Gate Suite Validation — {run_id}")
    lines.append("")
    if summary["duplicates"]:
        lines.append("## ⚠️ Duplicates")
        for d in summary["duplicates"]:
            lines.append(f"- Modes: **{', '.join(d['modes'])}**")
        lines.append("")
    lines.append("## Experiments (recomputed from scored JSONLs)")
    lines.append("")
    for exp in summary["experiments"]:
        mode = exp["mode"]
        rep = exp["report"]
        rec = exp.get("recomputed_eval") or {}
        if not rec:
            lines.append(f"- **{mode}**: (missing scored files) — check errors above.")
            continue
        lines.append(f"### {mode}")
        lines.append(f"- tau_cal: `{rec.get('tau'):.6f}`")
        lines.append(f"- EVAL FAR: `{fmt_pct(rec.get('far'))}`   | TPR: `{fmt_pct(rec.get('tpr'))}`   | AUC: `{rec.get('auc'):.3f}`")
        lines.append(f"- POS mean±std: `{rec['pos_stats']['mean']:.4f} ± {rec['pos_stats']['std']:.4f}`")
        lines.append(f"- NEG mean±std: `{rec['neg_stats']['mean']:.4f} ± {rec['neg_stats']['std']:.4f}`")
        lines.append(f"- Overlap (NEG <= POS median): `{fmt_pct(rec.get('overlap_neg_le_pos_p50'))}`")
        lines.append("")
    if args.cache_db and summary["cache_db"]:
        ci = summary["cache_db"]
        lines.append("## Cache DB")
        lines.append(f"- cache_db: `{ci.get('cache_db')}`")
        lines.append(f"- model: `{ci.get('model')}`")
        for k in ("resolved_total", "resolved_ok", "resolved_fail", "embeddings_total", "embeddings_for_model", "embeddings_missing_resolved_row"):
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

    # print console summary
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
