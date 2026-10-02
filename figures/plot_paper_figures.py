#!/usr/bin/env python3
"""Render colored CELL-ID plots with black text from frozen measured results.

Run from the project root: python figures/plot_paper_figures.py
Use --verify-only to check the saved files without writing outputs.
Only figures/paper/ is written. Existing nonidentical outputs are preserved by
refusing a rerun unless --overwrite is explicitly supplied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
sys.dont_write_bytecode = True

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, __version__ as pillow_version
import pymupdf as fitz

from figstyle import _text_collisions, panel_labels, use_style

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/paper"
WIDTH, DPI = 6.6, 600
SIZES = {"budget_curves": (WIDTH, 2.6), "simple_baseline": (WIDTH, 2.2)}
POLICIES = ("random", "trace-uncertainty", "D-optimal", "V-optimal")
LABELS = {"random": "Random", "trace-uncertainty": "Uncertainty",
          "D-optimal": "D-optimal", "V-optimal": "V-optimal"}
STYLES = {
    "random": dict(color="#777777", marker="o", linestyle=(0, (5, 2)), markersize=3.6),
    "trace-uncertainty": dict(color="#E69F00", marker="^", linestyle=(0, (1, 1.4)), markersize=5.2),
    "D-optimal": dict(color="#0072B2", marker="s", linestyle=(0, (4, 1.4, 1, 1.4)), markersize=4.2),
    "V-optimal": dict(color="#009E73", marker="D", linestyle="-", markersize=3.0),
}
CONTEXT_STYLE = dict(color="#CC79A7", marker="s", linestyle="-", markersize=3.8)
TABLES = {"curve_source_data.csv": ["metric", "policy", "budget", "mean", "lo", "hi"],
          "simple_baseline_source_data.csv": ["method", "budget", "mean", "lo", "hi"]}
SOURCES = (
    "results/pilot/main/per_context.csv", "results/pilot/context_shift/per_context.csv",
    "figures/figstyle.py",
)
OUTPUTS = tuple(OUT / f"{stem}.{fmt}" for stem in SIZES for fmt in ("png", "pdf", "svg"))
OUTPUTS += tuple(OUT / name for name in TABLES)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate():
    """Preserve the original pandas read/pivot/order/NumPy reductions.

    The per-context inputs already average random seeds. No seed is selected,
    no context is excluded, and no rounding, smoothing, or jitter is applied.
    """
    data = pd.read_csv(ROOT / SOURCES[0])
    shift = pd.read_csv(ROOT / SOURCES[1])
    contexts = sorted(data.context.unique())
    budgets = sorted(data.budget.unique())
    n = len(contexts)
    assert set(POLICIES).issubset(data.policy.unique())
    assert 5 in budgets
    assert set(shift.context.unique()) == set(contexts)
    assert set(shift.budget.unique()) == set(budgets)
    assert data.loc[data.policy.eq("random"), "n_seeds"].eq(20).all()
    assert shift.n_seeds.eq(20).all()
    resamples = np.random.default_rng(77).integers(0, n, size=(2000, n))

    def summarize(frame, metric):
        a = frame.pivot(index="context", columns="budget", values=metric).reindex(
            index=contexts, columns=budgets).to_numpy()
        assert np.isfinite(a).all(), f"Missing {metric} values"
        mean = a.mean(axis=0)
        lo, hi = np.quantile(a[resamples].mean(axis=1), [0.025, 0.975], axis=0)
        return mean, lo, hi

    table = []
    for metric in ("normalized_mse", "pearson_mean"):
        for policy in POLICIES:
            mean, lo, hi = summarize(data.loc[data.policy.eq(policy)], metric)
            for k, m, l, h in zip(budgets, mean, lo, hi):
                table.append(dict(metric=metric, policy=policy, budget=int(k),
                                  mean=float(m), lo=float(l), hi=float(h)))
    simple_table = []
    for frame, label in (
        (data[data.policy.eq("random")], "Latent model + random probes"),
        (shift, "Context offset (exploratory)"),
    ):
        mean, lo, hi = summarize(frame, "normalized_mse")
        for k, m, l, h in zip(budgets, mean, lo, hi):
            simple_table.append(dict(method=label, budget=int(k), mean=float(m),
                                     lo=float(l), hi=float(h)))
    tables = {"curve_source_data.csv": pd.DataFrame(table),
              "simple_baseline_source_data.csv": pd.DataFrame(simple_table)}
    for name, frame in tables.items():
        assert frame.columns.tolist() == TABLES[name]
    return tables, contexts, [int(k) for k in budgets]


def configure_style():
    use_style()
    mpl.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "text.color": "black", "axes.labelcolor": "black", "axes.edgecolor": "black",
        "xtick.color": "black", "ytick.color": "black", "legend.labelcolor": "black",
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "savefig.edgecolor": "white", "savefig.dpi": DPI, "figure.dpi": 150,
        "axes.prop_cycle": mpl.cycler(color=[STYLES[p]["color"] for p in POLICIES]),
        "grid.color": "#dddddd", "grid.linewidth": 0.4,
        "pdf.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "CELL-ID-paper",
        "legend.handlelength": 2.8, "legend.columnspacing": 1.7,
    })


def save_plot(fig, stem):
    fig.canvas.draw()
    problems = _text_collisions(fig)
    assert not problems, f"{stem}: {problems}"
    assert all(not ax.get_title() for ax in fig.axes)
    assert all(text.get_fontsize() >= 8 for text in fig.findobj(mpl.text.Text)
               if text.get_visible() and text.get_text().strip())
    for fmt in ("pdf", "svg", "png"):
        metadata = ({"CreationDate": None, "ModDate": None} if fmt == "pdf"
                    else {"Date": None} if fmt == "svg" else None)
        fig.savefig(OUT / f"{stem}.{fmt}", format=fmt, dpi=DPI, metadata=metadata)
    plt.close(fig)


def render(tables, budgets):
    configure_style()
    curves = tables["curve_source_data.csv"]
    fig, axes = plt.subplots(1, 2, figsize=SIZES["budget_curves"])
    fig.subplots_adjust(left=0.125, right=0.985, bottom=0.19, top=0.81, wspace=0.36)
    for ax, metric, ylabel in zip(axes, ("normalized_mse", "pearson_mean"),
                                 ("Normalized MSE\n(lower is better)", "Mean signature Pearson r")):
        for policy in POLICIES:
            values = curves[curves.metric.eq(metric) & curves.policy.eq(policy)]
            style = STYLES[policy]
            ax.plot(values.budget, values["mean"], label=LABELS[policy], **style,
                    linewidth=1.15, markerfacecolor="white", markeredgewidth=0.8)
            if policy in ("random", "V-optimal"):
                ax.fill_between(values.budget, values.lo, values.hi,
                                color=style["color"], alpha=0.10, linewidth=0)
        if metric == "normalized_mse":
            ax.axhline(1, color="#888888", linestyle=":", linewidth=1, zorder=0)
        ax.set_xlabel("Measured probes (count)")
        ax.set_ylabel(ylabel)
        ax.set_xticks([0, 5, 10, 20])
        ax.set_xlim(-0.5, max(budgets) + 0.5)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.54, 0.992),
               ncol=4, frameon=False)
    panel_labels(axes)
    save_plot(fig, "budget_curves")

    simple = tables["simple_baseline_source_data.csv"]
    fig, ax = plt.subplots(figsize=SIZES["simple_baseline"])
    fig.subplots_adjust(left=0.125, right=0.985, bottom=0.225, top=0.96)
    for label, style in (
        ("Latent model + random probes", STYLES["random"]),
        ("Context offset (exploratory)", CONTEXT_STYLE),
    ):
        values = simple[simple.method.eq(label)]
        ax.plot(values.budget, values["mean"], label=label, **style,
                linewidth=1.25, markerfacecolor="white", markeredgewidth=0.8)
        ax.fill_between(values.budget, values.lo, values.hi,
                        color=style["color"], alpha=0.12, linewidth=0)
    ax.axhline(1, color="#888888", linestyle=":", linewidth=1, zorder=0)
    ax.set_xlabel("Measured probes (count)")
    ax.set_ylabel("Normalized MSE\n(lower is better)")
    ax.set_xticks([0, 1, 3, 5, 10, 20])
    ax.set_xlim(-0.5, 20.5)
    ax.legend(loc="upper right", frameon=False)
    save_plot(fig, "simple_baseline")


def audit_svg_text(svg):
    """Resolve inherited SVG text paint independently of matplotlib settings."""
    sizes = []

    def visit(node, inherited, within_text=False):
        style = dict(inherited)
        style.update({key: node.attrib[key] for key in ("fill", "stroke", "font-size")
                      if key in node.attrib})
        style.update(dict(re.findall(r"([\w-]+)\s*:\s*([^;]+)", node.attrib.get("style", ""))))
        within_text = within_text or node.tag.rsplit("}", 1)[-1] == "text"
        if within_text and (node.text or "").strip():
            assert mpl.colors.to_rgb(style["fill"]) == (0, 0, 0), "Nonblack SVG text fill"
            assert style["stroke"] == "none" or mpl.colors.to_rgb(style["stroke"]) == (0, 0, 0), "Nonblack SVG text stroke"
            size = float(re.fullmatch(r"([\d.]+)(?:px|pt)?", style["font-size"])[1])
            assert size >= 8, "SVG text below 8 pt"
            sizes.append(size)
        for child in node:
            visit(child, style, within_text)

    visit(svg, {"fill": "black", "stroke": "none"})
    assert sizes, "No SVG text found"
    return min(sizes)


def verify_outputs(tables):
    # Disable ICC conversion so encoded PDF RGB colors are inspected directly.
    fitz.TOOLS.set_icc(False)
    checks = {"source_tables": {}, "plots": {}}
    for name, expected in tables.items():
        # Round-trip parsing avoids a CSV parser changing a serialized final bit.
        actual = pd.read_csv(OUT / name, float_precision="round_trip")
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        assert (OUT / name).read_bytes() == expected.to_csv(index=False).encode()
        difference = np.max(np.abs(actual[["mean", "lo", "hi"]].to_numpy()
                                   - expected[["mean", "lo", "hi"]].to_numpy()))
        checks["source_tables"][name] = {"rows": len(actual), "columns": actual.columns.tolist(),
                                        "reference": "recomputed directly from measured per-context result CSVs",
                                        "exact_match": True, "byte_identical": True,
                                        "max_abs_numeric_difference": float(difference)}
    for stem, size in SIZES.items():
        required_colors = (["#777777", "#E69F00", "#0072B2", "#009E73"]
                           if stem == "budget_curves" else ["#777777", "#CC79A7"])
        expected_pixels = tuple(round(v * DPI) for v in size)
        with Image.open(OUT / f"{stem}.png") as image:
            assert image.mode in ("RGB", "RGBA") and image.size == expected_pixels
            png_mode = image.mode
            rgb = np.asarray(image.convert("RGB"))
            np.testing.assert_allclose(image.info["dpi"], [DPI, DPI], atol=0.02, rtol=0)
            png_counts = {color: int(np.all(rgb == np.rint(np.array(mpl.colors.to_rgb(color)) * 255)
                                                   .astype(np.uint8), axis=-1).sum())
                          for color in required_colors}
            assert all(count > 0 for count in png_counts.values()), png_counts
        svg = ET.parse(OUT / f"{stem}.svg").getroot()
        svg_size = [float(svg.attrib[k].removesuffix("pt")) / 72 for k in ("width", "height")]
        np.testing.assert_allclose(svg_size, size, rtol=0, atol=1e-12)
        minimum_svg_font = audit_svg_text(svg)
        svg_text = (OUT / f"{stem}.svg").read_text()
        strokes = set(re.findall(r"stroke:\s*(#[0-9a-f]{6})", svg_text.lower()))
        assert {color.lower() for color in required_colors}.issubset(strokes), strokes
        with fitz.open(OUT / f"{stem}.pdf") as document:
            assert len(document) == 1
            page = document[0]
            pdf_size = [page.rect.width / 72, page.rect.height / 72]
            np.testing.assert_allclose(pdf_size, size, rtol=0, atol=1e-6)
            spans = [span for block in page.get_text("dict")["blocks"] if "lines" in block
                     for line in block["lines"] for span in line["spans"] if span["text"].strip()]
            assert spans and min(span["size"] for span in spans) >= 8 - 1e-5
            assert all(span["color"] == 0 for span in spans), "Nonblack PDF text"
            colors = [drawing["color"] for drawing in page.get_drawings()
                      if drawing["color"] is not None]
            for color in required_colors:
                assert any(np.allclose(actual, mpl.colors.to_rgb(color), rtol=0, atol=1e-6)
                           for actual in colors), (stem, color)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
            rgb = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, 3)
            assert np.any(np.ptp(rgb, axis=-1) > 0), "PDF renders without color"
            checks["plots"][stem] = {"inches": list(size), "pixels": list(expected_pixels),
                                     "png_mode": png_mode, "dpi": DPI,
                                     "required_colors_in_png_pdf_svg": required_colors,
                                     "png_exact_color_pixel_counts": png_counts,
                                     "pdf_color_audit": "RGB vector strokes and non-grayscale rendered pixels; ICC disabled",
                                     "minimum_pdf_font_pt": min(span["size"] for span in spans),
                                     "minimum_svg_font_pt": minimum_svg_font,
                                     "pdf_text_black": True, "svg_text_black": True,
                                     "pdf_fonts": sorted({span["font"] for span in spans})}
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly regenerate paper assets")
    args = parser.parse_args()
    hashes = {name: sha256(ROOT / name) for name in SOURCES}
    tables, contexts, budgets = aggregate()
    if not args.verify_only:
        expected = [*OUTPUTS, OUT / "provenance.json"]
        existing = [str(path.relative_to(ROOT)) for path in expected if path.exists()]
        if existing and not args.overwrite:
            raise FileExistsError(f"Preserving existing outputs: {existing}; use --verify-only or --overwrite")
        OUT.mkdir(exist_ok=True)
        for name, table in tables.items():
            table.to_csv(OUT / name, index=False)
        render(tables, budgets)
    checks = verify_outputs(tables)
    assert hashes == {name: sha256(ROOT / name) for name in SOURCES}, "Source modified during run"
    provenance = {
        "script": "figures/plot_paper_figures.py", "script_sha256": sha256(Path(__file__)),
        "command": " ".join([sys.executable, "figures/plot_paper_figures.py", *sys.argv[1:]]),
        "source_sha256": hashes,
        "aggregation": {"contexts": contexts, "budgets": budgets, "bootstrap_resamples": 2000,
                        "bootstrap_seed": 77, "context_order": "sorted identifier",
                        "interval": "95% percentile bootstrap over held-out contexts; descriptive because training folds overlap",
                        "random": "20 seeds already averaged per context in source CSVs",
                        "main_bands": ["random", "V-optimal"], "simple_bands": "both methods",
                        "exclusions": "none within plotted policies/metrics; unadapted policy is a reference line only",
                        "rounding_smoothing_jitter": "none"},
        "presentation": {"width_inches": WIDTH, "minimum_plot_font_pt": 8,
                          "background": "white", "text": "black", "titles_captions": "none in plots",
                          "line_styles": STYLES, "context_offset_style": CONTEXT_STYLE,
                          "overlap": "retained at exact measured coordinates",
                          "conceptual_images": "not generated, converted, read or hashed by this plot renderer"},
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__, "matplotlib": mpl.__version__, "pillow": pillow_version,
                     "pymupdf": fitz.VersionBind},
        "verification": checks,
        "output_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sorted(OUTPUTS)},
    }
    if args.verify_only:
        saved = json.loads((OUT / "provenance.json").read_text())
        assert saved["source_sha256"] == hashes
        assert saved["script_sha256"] == provenance["script_sha256"]
        assert saved["output_sha256"] == provenance["output_sha256"]
    else:
        (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"status": "verified", "contexts": len(contexts), "budgets": budgets,
                      "source_tables": checks["source_tables"],
                      "plots": checks["plots"]}, indent=2))


if __name__ == "__main__":
    main()
