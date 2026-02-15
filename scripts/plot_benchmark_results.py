import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt


def main(results_json: Path) -> None:
    with results_json.open("r") as file:
        results = json.load(file)

    for wavelet_name in results["config"]["wavelets"]:
        wavelet_results = sorted(
            [r for r in results["results"] if r["wavelet"] == wavelet_name],
            key=lambda r: r["length"],
        )

        lengths = [r["length"] for r in wavelet_results]
        x = range(len(lengths))

        ptwt_mean = [r["ptwt"]["mean_s"] for r in wavelet_results]
        ptwt_std = [r["ptwt"]["std_s"] for r in wavelet_results]
        fbwp_mean = [r["fbwp"]["mean_s"] for r in wavelet_results]
        fbwp_std = [r["fbwp"]["std_s"] for r in wavelet_results]
        speedups = [r["speedup"] for r in wavelet_results]

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(x, ptwt_mean, "o-", label="ptwt", color="#d62728")
        ax.fill_between(
            x,
            [m - s for m, s in zip(ptwt_mean, ptwt_std)],
            [m + s for m, s in zip(ptwt_mean, ptwt_std)],
            alpha=0.2, color="#d62728",
        )

        ax.plot(x, fbwp_mean, "o-", label="fbwp", color="#1f77b4")
        ax.fill_between(
            x,
            [m - s for m, s in zip(fbwp_mean, fbwp_std)],
            [m + s for m, s in zip(fbwp_mean, fbwp_std)],
            alpha=0.2, color="#1f77b4",
        )

        for i, sp in enumerate(speedups):
            ax.annotate(
                f"{sp:.1f}x",
                (i, fbwp_mean[i]),
                textcoords="offset points", xytext=(0, -14),
                ha="center", fontsize=9, color="#1f77b4",
            )

        tick_labels = [
            f"{r['length']}\n(L={r['max_level']})" for r in wavelet_results
        ]
        ax.set_xticks(list(x))
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel("Signal length (max level)")
        ax.set_ylabel("Time (s)")
        ax.set_title(f"Wavelet packet transform — {wavelet_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out_path = results_json.parent / f"benchmark_{wavelet_name}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-json", "-I", type=Path, required=True
    )
    main(**vars(parser.parse_args()))
