# src/certum/reporting/build_duckdb.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import duckdb


def _exists(p: Optional[Path]) -> bool:
    return p is not None and Path(p).exists()


def build_duckdb_for_run(
    *,
    out_duckdb: Path,
    out_report: Path,
    out_pos_policies: Optional[Path],
    out_neg_policies: Optional[Path],
    out_pos_scored: Optional[Path],
    out_neg_scored: Optional[Path],
) -> Path:
    """
    Build a DuckDB database in the run directory containing:
      - policy_sweep (pos+neg policy sweep rows)
      - eval_scored  (pos+neg scored evaluation rows, if provided)
      - run_report   (the JSON report as a single-row table)
      - helpful views for analysis
    """

    out_duckdb = Path(out_duckdb)
    out_duckdb.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(out_duckdb))
    con.execute("PRAGMA threads=4")

    # -----------------------
    # policy_sweep table
    # -----------------------
    if _exists(out_pos_policies) and _exists(out_neg_policies):
        con.execute(
            """
            CREATE OR REPLACE TABLE policy_sweep AS
            SELECT * FROM read_json_auto(?, format='newline_delimited')
            UNION ALL
            SELECT * FROM read_json_auto(?, format='newline_delimited')
            """,
            [str(out_pos_policies), str(out_neg_policies)],
        )
    elif _exists(out_pos_policies):
        con.execute(
            "CREATE OR REPLACE TABLE policy_sweep AS SELECT * FROM read_json_auto(?, format='newline_delimited')",
            [str(out_pos_policies)],
        )
    elif _exists(out_neg_policies):
        con.execute(
            "CREATE OR REPLACE TABLE policy_sweep AS SELECT * FROM read_json_auto(?, format='newline_delimited')",
            [str(out_neg_policies)],
        )
    else:
        con.execute(
            "CREATE OR REPLACE TABLE policy_sweep AS SELECT NULL::VARCHAR AS run_id WHERE FALSE"
        )

    # -----------------------
    # eval_scored table (optional)
    # -----------------------
    if _exists(out_pos_scored) and _exists(out_neg_scored):
        con.execute(
            """
            CREATE OR REPLACE TABLE eval_scored AS
            SELECT * FROM read_json_auto(?, format='newline_delimited')
            UNION ALL
            SELECT * FROM read_json_auto(?, format='newline_delimited')
            """,
            [str(out_pos_scored), str(out_neg_scored)],
        )
    elif _exists(out_pos_scored):
        con.execute(
            "CREATE OR REPLACE TABLE eval_scored AS SELECT * FROM read_json_auto(?, format='newline_delimited')",
            [str(out_pos_scored)],
        )
    elif _exists(out_neg_scored):
        con.execute(
            "CREATE OR REPLACE TABLE eval_scored AS SELECT * FROM read_json_auto(?, format='newline_delimited')",
            [str(out_neg_scored)],
        )
    else:
        con.execute(
            "CREATE OR REPLACE TABLE eval_scored AS SELECT NULL::VARCHAR AS meta WHERE FALSE"
        )

    # -----------------------
    # run_report table (single row)
    # -----------------------
    if Path(out_report).exists():
        report_obj = json.loads(Path(out_report).read_text(encoding="utf-8"))
        con.execute("CREATE OR REPLACE TABLE run_report (report_json JSON)")
        con.execute("DELETE FROM run_report")
        con.execute("INSERT INTO run_report VALUES (?)", [json.dumps(report_obj)])
    else:
        con.execute("CREATE OR REPLACE TABLE run_report (report_json JSON)")
        con.execute("DELETE FROM run_report")

    # -----------------------
    # Helpful views
    # -----------------------

    # policy accept/review/reject rates by policy+split
    con.execute(
        """
        CREATE OR REPLACE VIEW v_policy_rates AS
        SELECT
        policy_name AS policy,
        split,
        count(*) AS n,
        avg(CASE WHEN verdict = 'accept' THEN 1.0 ELSE 0.0 END) AS accept_rate,
        avg(CASE WHEN verdict = 'review' THEN 1.0 ELSE 0.0 END) AS review_rate,
        avg(CASE WHEN verdict = 'reject' THEN 1.0 ELSE 0.0 END) AS reject_rate,
        any_value(tau_accept) AS tau_accept
        FROM policy_sweep
        GROUP BY 1,2
        """
    )

    # baseline tau from EnergyOnly rows (if present)
    con.execute(
        """
        CREATE OR REPLACE VIEW v_tau_energy AS
        SELECT any_value(tau_accept) AS tau_energy
        FROM policy_sweep
        WHERE policy_name LIKE '%EnergyOnly%'
        """
    )

    # false-accept slice: NEG that pass energy-only threshold
    con.execute(
        """
        CREATE OR REPLACE VIEW v_neg_false_accepts_energy_only AS
        SELECT *
        FROM policy_sweep
        WHERE split='neg'
          AND policy_name LIKE '%EnergyOnly%'
          AND energy <= (SELECT tau_energy FROM v_tau_energy)
        """
    )

    con.close()
    return out_duckdb
