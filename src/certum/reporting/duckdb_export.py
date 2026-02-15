# src/certum/reporting/duckdb_export.py
from __future__ import annotations

import json
from pathlib import Path

import duckdb


def export_run_to_duckdb(
    *,
    out_db: Path,
    report_json: Path,
    pos_policies_jsonl: Path | None,
    neg_policies_jsonl: Path | None,
    pos_scored_jsonl: Path | None = None,
    neg_scored_jsonl: Path | None = None,
) -> None:
    out_db.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(out_db))
    con.execute("PRAGMA threads=4;")

    # -------------------------
    # 1) runs + calibration
    # -------------------------
    report = json.loads(report_json.read_text(encoding="utf-8"))

    con.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id VARCHAR,
            model VARCHAR,
            far DOUBLE,
            neg_mode VARCHAR,
            seed INTEGER,
            cal_frac DOUBLE,
            n_total INTEGER
        );
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS calibration (
            run_id VARCHAR,
            tau_energy DOUBLE,
            tau_pr DOUBLE,
            tau_sensitivity DOUBLE,
            hard_negative_gap DOUBLE
        );
    """)

    con.execute("DELETE FROM runs WHERE run_id = ?;", [report["run_id"]])
    con.execute("DELETE FROM calibration WHERE run_id = ?;", [report["run_id"]])

    con.execute(
        "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?);",
        [
            report["run_id"],
            report.get("model"),
            float(report.get("far", 0.0)),
            report.get("neg_mode"),
            int(report.get("seed", 0)),
            float(report.get("cal_frac", 0.0)),
            int(report.get("n_total", 0)),
        ],
    )

    cal = report.get("calibration", {}) or {}
    con.execute(
        "INSERT INTO calibration VALUES (?, ?, ?, ?, ?);",
        [
            report["run_id"],
            float(cal.get("tau_energy", 0.0)),
            float(cal.get("tau_pr", 0.0)),
            float(cal.get("tau_sensitivity", 0.0)),
            float(cal.get("hard_negative_gap", 0.0)),
        ],
    )

    # -------------------------
    # 2) policy_sweep
    # -------------------------
    con.execute("""
        CREATE TABLE IF NOT EXISTS policy_sweep (
            run_id VARCHAR,
            split VARCHAR,
            sample_id VARCHAR,
            source_id VARCHAR,
            row_id VARCHAR,
            policy_name VARCHAR,
            policy_key VARCHAR,
            verdict VARCHAR,
            effectiveness DOUBLE,
            energy DOUBLE,
            explained DOUBLE,
            participation_ratio DOUBLE,
            sensitivity DOUBLE,
            alignment DOUBLE,
            sim_margin DOUBLE,
            tau_accept DOUBLE,
            effective_rank INTEGER,
            used_count INTEGER,
            sigma1_ratio DOUBLE,
            sigma2_ratio DOUBLE,
            entropy_rank DOUBLE,
            sim_top1 DOUBLE,
            sim_top2 DOUBLE,
            embedding_backend VARCHAR,
            claim_dim INTEGER,
            evidence_count INTEGER,
            energy_result JSON
        );
    """)

    # remove old rows for this run (if re-running)
    con.execute("DELETE FROM policy_sweep WHERE run_id = ?;", [report["run_id"]])

    def _ingest_policy_jsonl(path: Path):
        # DuckDB can auto-read newline-delimited JSON
        con.execute(
            """
            INSERT INTO policy_sweep
            SELECT
                run_id,
                split,
                sample_id,
                source_id,
                row_id,
                policy_name,
                policy_key,
                verdict,
                effectiveness,
                energy,
                explained,
                participation_ratio,
                sensitivity,
                alignment,
                sim_margin,
                tau_accept,
                effective_rank,
                used_count,
                sigma1_ratio,
                sigma2_ratio,
                entropy_rank,
                sim_top1,
                sim_top2,
                embedding_backend,
                claim_dim,
                evidence_count,
                to_json(energy_result) AS energy_result
            FROM read_json_auto(?, format='newline_delimited');
            """,
            [str(path)],
        )

    if pos_policies_jsonl and pos_policies_jsonl.exists():
        _ingest_policy_jsonl(pos_policies_jsonl)
    if neg_policies_jsonl and neg_policies_jsonl.exists():
        _ingest_policy_jsonl(neg_policies_jsonl)

    # Optional: create convenience view
    con.execute("""
        CREATE OR REPLACE VIEW policy_metrics AS
        SELECT
            policy_name,
            AVG(CASE WHEN split='pos' AND verdict='accept' THEN 1 ELSE 0 END) AS tpr,
            AVG(CASE WHEN split='neg' AND verdict='accept' THEN 1 ELSE 0 END) AS far,
            AVG(CASE WHEN split='pos' AND verdict='review' THEN 1 ELSE 0 END) AS pos_review_rate,
            AVG(CASE WHEN split='neg' AND verdict='review' THEN 1 ELSE 0 END) AS neg_review_rate
        FROM policy_sweep
        GROUP BY 1;
    """)

    con.close()
