import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import duckdb
import numpy as np


def _col_exists(con: duckdb.DuckDBPyConnection, table: str, col: str) -> bool:
    rows = con.execute(f"PRAGMA table_info('{table}')").fetchall()
    cols = {r[1] for r in rows}  # (cid, name, type, ...)
    return col in cols


def _pick_mode_column(con: duckdb.DuckDBPyConnection) -> Optional[str]:
    # If your policy_sweep includes mode/neg_mode, use it. Otherwise return None.
    for c in ("mode", "neg_mode"):
        if _col_exists(con, "policy_sweep", c):
            return c
    return None


def _choose_baseline_policy(con: duckdb.DuckDBPyConnection) -> str:
    # Prefer adaptive if present, else first policy_name.
    pols = [r[0] for r in con.execute("SELECT DISTINCT policy_name FROM policy_sweep").fetchall()]
    if "Adaptive" in pols:
        return "Adaptive"
    if "adaptive" in pols:
        return "adaptive"
    if "energy_only" in pols:
        return "energy_only"
    return sorted(pols)[0]


def _get_tau_energy(con: duckdb.DuckDBPyConnection) -> Optional[float]:
    # Prefer a runs table if you have one; otherwise try calibration table/view;
    # otherwise fall back to median tau_accept across policies that have it.
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}

    if "runs" in tables and _col_exists(con, "runs", "tau_energy"):
        row = con.execute("SELECT any_value(tau_energy) FROM runs").fetchone()
        return float(row[0]) if row and row[0] is not None else None

    if "calibration" in tables:
        # adjust if your schema differs
        for c in ("tau_energy", "tau"):
            if _col_exists(con, "calibration", c):
                row = con.execute(f"SELECT any_value({c}) FROM calibration").fetchone()
                return float(row[0]) if row and row[0] is not None else None

    # fallback: median tau_accept from policy_sweep (ignores axis-only policies)
    if _col_exists(con, "policy_sweep", "tau_accept"):
        row = con.execute(
            "SELECT median(tau_accept) FROM policy_sweep WHERE tau_accept IS NOT NULL"
        ).fetchone()
        return float(row[0]) if row and row[0] is not None else None

    return None


def _energy_band(
    con: duckdb.DuckDBPyConnection,
    baseline_policy: str,
    tau_energy: float,
    min_per_class: int = 200,
    w_max: float = 0.20,
) -> Tuple[float, int, int]:
    """
    Find smallest band width w such that within |energy - tau| <= w
    we have at least min_per_class in BOTH pos and neg.
    """
    widths = [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.12, 0.15, w_max]
    for w in widths:
        q = """
        SELECT
          sum(split='pos')::INT AS n_pos,
          sum(split='neg')::INT AS n_neg
        FROM policy_sweep
        WHERE policy_name = ?
          AND abs(energy - ?) <= ?
        """
        n_pos, n_neg = con.execute(q, [baseline_policy, tau_energy, w]).fetchone()
        n_pos = int(n_pos or 0)
        n_neg = int(n_neg or 0)
        if n_pos >= min_per_class and n_neg >= min_per_class:
            return float(w), n_pos, n_neg

    # return widest even if unbalanced; still useful diagnostics
    w = float(widths[-1])
    n_pos, n_neg = con.execute(
        """
        SELECT sum(split='pos')::INT, sum(split='neg')::INT
        FROM policy_sweep
        WHERE policy_name = ?
          AND abs(energy - ?) <= ?
        """,
        [baseline_policy, tau_energy, w],
    ).fetchone()
    return w, int(n_pos or 0), int(n_neg or 0)


def build_views(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE OR REPLACE VIEW v_policy_rates AS
        SELECT
          policy_name,
          split,
          count(*)::BIGINT AS n,
          avg(verdict='accept')::DOUBLE AS accept_rate,
          avg(verdict='review')::DOUBLE AS review_rate,
          avg(verdict='reject')::DOUBLE AS reject_rate,
          any_value(tau_accept) AS tau_accept
        FROM policy_sweep
        GROUP BY 1,2
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW v_policy_leaderboard AS
        SELECT
          pos.policy_name,
          pos.n AS n_pos,
          neg.n AS n_neg,
          pos.accept_rate AS pos_accept,
          pos.review_rate AS pos_review,
          pos.reject_rate AS pos_reject,
          neg.accept_rate AS neg_accept,
          neg.review_rate AS neg_review,
          neg.reject_rate AS neg_reject,
          (pos.accept_rate - neg.accept_rate) AS separation,
          any_value(pos.tau_accept) AS tau_accept
        FROM (SELECT * FROM v_policy_rates WHERE split='pos') pos
        JOIN (SELECT * FROM v_policy_rates WHERE split='neg') neg
          USING (policy_name)
        ORDER BY neg_accept ASC, pos_accept DESC
        """
    )


def write_markdown(con: duckdb.DuckDBPyConnection, out_md: Path, meta: dict) -> None:
    top = con.execute("SELECT * FROM v_policy_leaderboard LIMIT 50").fetchdf()

    # Pareto frontier (simple filter)
    rows = top.to_dict(orient="records")
    pareto = []
    for r in rows:
        dominated = False
        for o in rows:
            if (o["neg_accept"] <= r["neg_accept"]) and (o["pos_accept"] >= r["pos_accept"]):
                if (o["neg_accept"] < r["neg_accept"]) or (o["pos_accept"] > r["pos_accept"]):
                    dominated = True
                    break
        if not dominated:
            pareto.append(r)

    def fmt(x: float) -> str:
        if x is None:
            return "NA"
        return f"{float(x):.4f}"

    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Certum Policy Sweep Report\n\n")
        f.write("## Run metadata\n\n")
        f.write("```json\n")
        f.write(json.dumps(meta, indent=2))
        f.write("\n```\n\n")

        f.write("## Policy leaderboard (sorted by lowest NEG accept, then highest POS accept)\n\n")
        f.write("| policy | pos_accept | pos_review | neg_accept | neg_review | separation |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for r in rows[:25]:
            f.write(
                f"| {r['policy_name']} | {fmt(r['pos_accept'])} | {fmt(r['pos_review'])} | "
                f"{fmt(r['neg_accept'])} | {fmt(r['neg_review'])} | {fmt(r['separation'])} |\n"
            )

        f.write("\n## Pareto-optimal policies (no policy is strictly better on both risk & coverage)\n\n")
        f.write("| policy | pos_accept | neg_accept | separation |\n")
        f.write("|---|---:|---:|---:|\n")
        for r in pareto:
            f.write(
                f"| {r['policy_name']} | {fmt(r['pos_accept'])} | {fmt(r['neg_accept'])} | {fmt(r['separation'])} |\n"
            )


def main():
    run_dirs = sorted(Path("artifacts/runs").glob("*"))
    if not run_dirs:
        raise RuntimeError("No runs found.")

    default_run = run_dirs[-1]

    ap = argparse.ArgumentParser()
    ap.add_argument("--duckdb", type=Path, default=default_run / "certum_results.duckdb")
    ap.add_argument("--out_dir", type=Path, default=default_run)
    ap.add_argument("--min_band", type=int, default=200)
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(args.duckdb), read_only=False)

    build_views(con)

    baseline = _choose_baseline_policy(con)
    tau = _get_tau_energy(con)

    meta = {
        "duckdb": str(args.duckdb),
        "baseline_policy": baseline,
        "tau_energy": tau,
    }

    if tau is not None:
        w, n_pos, n_neg = _energy_band(con, baseline, tau, min_per_class=args.min_band)
        meta["energy_band_width"] = w
        meta["band_n_pos"] = n_pos
        meta["band_n_neg"] = n_neg

        # Band-limited leaderboard
        con.execute("DROP VIEW IF EXISTS v_policy_rates_band")
        con.execute("DROP VIEW IF EXISTS v_policy_leaderboard_band")
        con.execute(
            """
            CREATE OR REPLACE VIEW v_policy_rates_band AS
            SELECT
              policy_name,
              split,
              count(*)::BIGINT AS n,
              avg(verdict='accept')::DOUBLE AS accept_rate,
              avg(verdict='review')::DOUBLE AS review_rate,
              avg(verdict='reject')::DOUBLE AS reject_rate,
              any_value(tau_accept) AS tau_accept
            FROM policy_sweep
            WHERE abs(energy - ?) <= ?
            GROUP BY 1,2
            """,
            [tau, w],
        )
        con.execute(
            """
            CREATE OR REPLACE VIEW v_policy_leaderboard_band AS
            SELECT
              pos.policy_name,
              pos.accept_rate AS pos_accept,
              neg.accept_rate AS neg_accept,
              (pos.accept_rate - neg.accept_rate) AS separation
            FROM (SELECT * FROM v_policy_rates_band WHERE split='pos') pos
            JOIN (SELECT * FROM v_policy_rates_band WHERE split='neg') neg
              USING (policy_name)
            ORDER BY neg_accept ASC, pos_accept DESC
            """
        )

        band = con.execute("SELECT * FROM v_policy_leaderboard_band LIMIT 25").fetchdf()
        band.to_csv(out_dir / "policy_leaderboard_band.csv", index=False)

    # Write main outputs
    (out_dir / "policy_leaderboard.csv").write_text(
        con.execute("SELECT * FROM v_policy_leaderboard").fetchdf().to_csv(index=False),
        encoding="utf-8",
    )
    write_markdown(con, out_dir / "policy_report.md", meta)

    con.close()
    print(f"✅ Wrote: {out_dir / 'policy_report.md'}")
    print(f"✅ Wrote: {out_dir / 'policy_leaderboard.csv'}")
    if tau is not None:
        print(f"✅ Wrote: {out_dir / 'policy_leaderboard_band.csv'}")


if __name__ == "__main__":
    main()
