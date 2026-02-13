# scripts/run_certum.py

import argparse
from certum.runner import CertumRunner


def main():
    ap = argparse.ArgumentParser()
    # (same arguments as before)
    args = ap.parse_args()

    runner = CertumRunner()

    runner.run(
        kind=args.kind,
        in_path=args.in_path,
        model_name=args.model,
        cache_db=args.cache_db,
        embedding_db=args.embedding_db,
        regime=args.regime,
        far=args.far,
        cal_frac=args.cal_frac,
        n=args.n,
        seed=args.seed,
        neg_mode=args.neg_mode,
        neg_offset=args.neg_offset,
        out_report=args.out_report,
        out_pos_scored=args.out_pos_scored,
        out_neg_scored=args.out_neg_scored,
        plot_png=args.plot_png,
    )


if __name__ == "__main__":
    main()
