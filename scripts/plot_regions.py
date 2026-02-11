import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def load_points(path: Path):
    pts = []
    print(f"📥 Loading points from: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            # print(f"Loaded point: \n{json.dumps(r, indent=2)}")
            e = r["energy"]["energy"] if "energy" in r and isinstance(r["energy"], dict) else r["energy_result"]["energy"]
            d = r["difficulty_value"]
            v = r["verdict"]
            pts.append((d, e, v))
    return pts

def plot(points, title: str, *, difficulty_low=0.40, difficulty_high=0.75, tau_accept=None, tau_review=None, margin_frac=0.10):
    # Split by verdict
    xs_a, ys_a = [], []
    xs_r, ys_r = [], []
    xs_x, ys_x = [], []

    for d, e, v in points:
        if v == "accept":
            xs_a.append(d); ys_a.append(e)
        elif v == "review":
            xs_r.append(d); ys_r.append(e)
        else:
            xs_x.append(d); ys_x.append(e)

    plt.figure()
    plt.scatter(xs_a, ys_a, label="accept", s=8)
    plt.scatter(xs_r, ys_r, label="review", s=8)
    plt.scatter(xs_x, ys_x, label="reject", s=8)

    # Difficulty bands
    plt.axvline(difficulty_low)
    plt.axvline(difficulty_high)

    # Energy thresholds (optional)
    if tau_accept is not None:
        plt.axhline(tau_accept, color="black")
        margin = margin_frac * tau_accept
        plt.axhline(tau_accept - margin, linestyle="--", color="gray")
        plt.axhline(tau_accept + margin, linestyle="--", color="gray")

    if tau_review is not None:
        plt.axhline(tau_review, color="red")

    plt.title(title)
    plt.xlabel("difficulty")
    plt.ylabel("energy")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_dirs = sorted(Path("artifacts/runs").glob("*"))
    if not run_dirs:
        raise RuntimeError("No runs found in artifacts/runs/")
    run_dir = run_dirs[-1]
    print(f"📊 Loading data from: {run_dir}")


    # pos = load_points(Path(f"{run_dir}/pos_hard_mined_v2.jsonl"))
    # neg = load_points(Path(f"{run_dir}/neg_hard_mined_v2.jsonl"))
    pos = load_points(Path(f"{run_dir}/pos_deranged.jsonl"))
    neg = load_points(Path(f"{run_dir}/neg_deranged.jsonl"))


    neg_difficulties = [d for d, e, v in neg]
    neg_energies = [e for d, e, v in neg]

    print("Energy–Difficulty correlation:", np.corrcoef(neg_energies, neg_difficulties)[0,1])

    print("Mean difficulty (accepted neg):",
        np.mean([d for d,e,v in neg if v=="accept"]))

    print("Mean difficulty (rejected neg):",
        np.mean([d for d,e,v in neg if v=="reject"]))


    # If you have tau in a report.json, load it and pass here.
    tau_accept = 0.5105144947767257
    tau_review = 0.6381431184709072
    plot(pos, "POS: energy vs difficulty", tau_accept=tau_accept, tau_review=tau_review)
    plot(neg, "NEG: energy vs difficulty", tau_accept=tau_accept, tau_review=tau_review)
