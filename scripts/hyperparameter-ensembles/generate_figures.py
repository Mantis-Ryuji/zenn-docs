"""Regenerate the article's illustrative plot and the supplied paper figures.

Run from the repository root:
    python scripts/hyperparameter-ensembles/generate_figures.py --only example
    python scripts/hyperparameter-ensembles/generate_figures.py --only paper

Dependencies and source/figure mappings are documented in README.md.
The illustrative probabilities are chosen examples, not experimental results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper" / "Hyperparameter Ensembles for Robustness and Uncertainty Quantification"
OUTPUT_DIR = ROOT / "images" / "zenn" / "hyperparameter-ensembles"
MAX_BYTES = 3_000_000
PAPER_IMAGES = {
    "blue_red_figure_init_vs_lambdas.pdf": "diversity-grid.png",
    "example_of_lower_upper_lambda_dynamic.pdf": "learned-ranges.png",
    "hparam_ens_cifar100_test_acc.pdf": "cifar100-accuracy.png",
    "hparam_ens_cifar100_test_ce.pdf": "cifar100-nll.png",
    "corruption_cifar10_accuracy.pdf": "corruption-accuracy.png",
    "corruption_cifar10_loss.pdf": "corruption-nll.png",
}


def check_image(path: Path) -> None:
    from PIL import Image

    size = path.stat().st_size
    if size > MAX_BYTES:
        raise ValueError(f"{path.name}: {size:,} bytes exceeds {MAX_BYTES:,}")
    with Image.open(path) as image:
        image.load()
        print(f"{path.name}: {image.width} x {image.height}, {size:,} bytes")


def render_example(output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Each coordinate is the true-class probability for a different example.
    # These vectors therefore need not sum to one.
    a = np.array([0.90, 0.30])
    b = np.array([0.85, 0.30])
    c = np.array([0.40, 0.60])
    ab, ac = (a + b) / 2, (a + c) / 2
    points = {"A": a, "B": b, "C": c, "(A+B)/2": ab, "(A+C)/2": ac}
    nll = {name: float(-np.log(point).mean()) for name, point in points.items()}
    if not (nll["A"] < nll["B"] < nll["C"] and nll["(A+C)/2"] < nll["A"]):
        raise ValueError("The illustrative example no longer has the stated ordering.")

    with plt.rc_context({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 16,
        "axes.edgecolor": "#697586",
        "text.color": "#172b4d",
        "axes.labelcolor": "#172b4d",
        "xtick.color": "#42526e",
        "ytick.color": "#42526e",
        "savefig.facecolor": "white",
    }):
        fig, ax = plt.subplots(figsize=(9, 6.5), layout="constrained")
        grid_x, grid_y = np.meshgrid(
            np.linspace(0.30, 0.99, 500), np.linspace(0.21, 0.70, 500)
        )
        losses = -0.5 * (np.log(grid_x) + np.log(grid_y))
        contours = ax.contour(
            grid_x, grid_y, losses,
            levels=[0.55, 0.60, 0.65, 0.70, 0.80, 0.90, 1.00],
            colors="#bac3ce", linewidths=0.9, zorder=0,
        )
        ax.clabel(contours, fmt="%.2f", fontsize=10, colors="#67788d")
        ax.plot(*np.vstack([a, b]).T, color="#dc8645", linewidth=3, zorder=2)
        ax.plot(*np.vstack([a, c]).T, color="#148579", linewidth=2, zorder=2)

        labels = {
            "A": ((0.935, 0.385), "#253c66", "o"),
            "B": ((0.790, 0.355), "#8d4f22", "o"),
            "C": ((0.380, 0.650), "#126c62", "o"),
            "(A+B)/2": ((0.740, 0.248), "#b85d1c", "D"),
            "(A+C)/2": ((0.675, 0.520), "#087568", "D"),
        }
        for name, point in points.items():
            label_at, color, marker = labels[name]
            ax.scatter(
                *point, s=100 if marker == "o" else 125, c=color, marker=marker,
                edgecolors="white", linewidths=1.2, zorder=4,
            )
            ax.annotate(
                f"{name}\nNLL {nll[name]:.3f}", point, xytext=label_at,
                ha="center", va="center", color=color, fontsize=11,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.93, "pad": 2},
                arrowprops={"arrowstyle": "-", "color": color, "lw": 0.9},
                zorder=5,
            )
        ax.annotate(
            "Lower NLL", xy=(0.95, 0.65), xytext=(0.78, 0.59),
            color="#67788d", fontsize=11,
            arrowprops={"arrowstyle": "->", "color": "#67788d"},
        )
        ax.set(
            xlim=(0.30, 0.99), ylim=(0.21, 0.70),
            xlabel=r"Example 1: true-class probability $q_1$",
            ylabel=r"Example 2: true-class probability $q_2$",
            title="A useful partner can rank lower on its own",
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(
            0, -0.17,
            "Circles: individual models   |   Diamonds: probability averages\n"
            r"Contours: mean NLL $=-\frac{1}{2}\log(q_1q_2)$   |   Illustrative example",
            transform=ax.transAxes, fontsize=10, color="#52657a", va="top",
        )
        path = output_dir / "complementarity.png"
        fig.savefig(path, dpi=180, pil_kwargs={"optimize": True})
        plt.close(fig)
    print("Illustrative NLL:", ", ".join(f"{name}={loss:.6f}" for name, loss in nll.items()))
    check_image(path)


def render_paper(output_dir: Path, paper_dir: Path, dpi: int) -> None:
    import pypdfium2 as pdfium

    source_dir = paper_dir / "figures"
    sources = [source_dir / name for name in PAPER_IMAGES]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing source PDFs:\n" + "\n".join(missing))
    for source_name, destination_name in PAPER_IMAGES.items():
        source = source_dir / source_name
        document = pdfium.PdfDocument(source)
        try:
            if len(document) != 1:
                raise ValueError(f"{source.name}: expected a one-page figure PDF")
            page = document[0]
            try:
                bitmap = page.render(scale=dpi / 72)
                try:
                    with bitmap.to_pil().convert("RGB") as rendered:
                        destination = output_dir / destination_name
                        rendered.save(destination, optimize=True)
                finally:
                    bitmap.close()
            finally:
                page.close()
        finally:
            document.close()
        check_image(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("example", "paper", "all"), default="all")
    parser.add_argument("--paper-dir", type=Path, default=PAPER_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=200, help="Resolution of paper PDF renders")
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.only in ("example", "all"):
        render_example(args.output_dir)
    if args.only in ("paper", "all"):
        render_paper(args.output_dir, args.paper_dir, args.dpi)


if __name__ == "__main__":
    main()
