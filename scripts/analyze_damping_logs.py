#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Analyze VBD artificial-damping diagnostic logs.

Reads the output of run_damping_ablation.py (or individual diagnostics.npz
files) and produces:
  - summary CSV with one row per run
  - energy-vs-time plots per scene/config
  - cumulative truncation energy loss
  - R_tangent / R_normal ratio over time
  - number of truncated vertices over time
  - residual vs iteration (if per-iteration data available)
  - automated report of likely damping causes

Usage:
    # Analyze a whole ablation run:
    uv run scripts/analyze_damping_logs.py --metadata-csv runs/damping_ablation/metadata.csv

    # Analyze a single .npz file:
    uv run scripts/analyze_damping_logs.py --npz output/diagnostics.npz

    # Both: pick output directory for plots
    uv run scripts/analyze_damping_logs.py \\
        --metadata-csv runs/damping_ablation/metadata.csv \\
        --output-dir analysis/
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# NPZ helpers
# ---------------------------------------------------------------------------

def load_tag(npz, tag: str) -> dict[str, np.ndarray]:
    """Return all arrays whose key starts with ``<tag>/``."""
    prefix = f"{tag}/"
    return {k[len(prefix):]: npz[k] for k in npz.files if k.startswith(prefix)}


def get_scalar(d: dict, key: str) -> Optional[np.ndarray]:
    return d.get(key)


# ---------------------------------------------------------------------------
# Single-file analysis
# ---------------------------------------------------------------------------

def analyze_single(npz_path: str, output_dir: str, label: str = "") -> dict:
    """Compute summary metrics from one diagnostics.npz file."""
    if not os.path.exists(npz_path):
        return {"status": "missing", "path": npz_path}

    try:
        npz = np.load(npz_path, allow_pickle=False)
    except Exception as e:
        return {"status": "load_error", "error": str(e), "path": npz_path}

    os.makedirs(output_dir, exist_ok=True)
    tag_label = label.replace("/", "_").replace(" ", "_")

    summary: dict = {"path": npz_path, "status": "ok"}

    # --- Step-level tags ---
    for tag in ["before_prediction", "after_prediction", "after_solver", "end_of_step"]:
        d = load_tag(npz, tag)
        if not d:
            continue
        ke = get_scalar(d, "kinetic_energy")
        if ke is not None and len(ke) > 0:
            summary[f"{tag}/ke_initial"] = float(ke[0])
            summary[f"{tag}/ke_final"] = float(ke[-1])
            summary[f"{tag}/ke_ratio"] = float(ke[-1]) / max(float(ke[0]), 1e-30)
        te = get_scalar(d, "total_energy")
        if te is not None and len(te) > 0:
            summary[f"{tag}/total_energy_initial"] = float(te[0])
            summary[f"{tag}/total_energy_final"] = float(te[-1])

    # --- Truncation ---
    td = load_tag(npz, "truncation")
    if td:
        nt = td.get("n_truncated")
        tf = td.get("tangential_fraction")
        dke = td.get("delta_kinetic_energy")
        dte = td.get("delta_total_energy")
        rn = td.get("total_r_normal")
        rt = td.get("total_r_tangent")

        if nt is not None:
            summary["truncation/mean_n_truncated"] = float(nt.mean())
            summary["truncation/max_n_truncated"] = float(nt.max())
        if tf is not None:
            summary["truncation/mean_tangential_fraction"] = float(tf.mean())
        if dke is not None:
            summary["truncation/cumulative_delta_ke"] = float(dke.sum())
        if dte is not None:
            summary["truncation/cumulative_delta_total_energy"] = float(dte.sum())
        if rn is not None and rt is not None:
            total_rn = float(rn.sum())
            total_rt = float(rt.sum())
            summary["truncation/overall_tangential_fraction"] = (
                total_rt / max(total_rn + total_rt, 1e-30)
            )

    # --- Iteration-level (if present) ---
    it = load_tag(npz, "iterations")
    if it:
        ke_iter = it.get("kinetic_energy")
        res = it.get("residual_mean")
        iters = it.get("iter")
        if res is not None and iters is not None:
            # residual at last iteration per step
            summary["iterations/final_residual_mean"] = float(res[-1]) if len(res) > 0 else float("nan")

    # --- Plots ---
    _plot_single(npz, output_dir, tag_label)

    return summary


def _plot_single(npz, output_dir: str, label: str) -> None:
    """Generate plots for a single diagnostics file."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[analyze] matplotlib not available — skipping plots")
        return

    # 1. Energy vs substep (end_of_step)
    d_eos = load_tag(npz, "end_of_step")
    if d_eos:
        substep = d_eos.get("substep", np.arange(len(d_eos.get("kinetic_energy", []))))
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"Energy vs. substep  [{label}]")

        for ax, (key, title) in zip(
            axes.ravel(),
            [
                ("kinetic_energy", "Kinetic Energy"),
                ("stretch_energy", "Stretch Energy"),
                ("bending_energy", "Bending Energy"),
                ("total_energy", "Total Energy"),
            ],
        ):
            arr = d_eos.get(key)
            if arr is not None:
                ax.plot(substep[: len(arr)], arr)
                ax.set_xlabel("substep")
                ax.set_ylabel("J")
                ax.set_title(title)
                ax.grid(True, alpha=0.4)
        plt.tight_layout()
        out = os.path.join(output_dir, f"{label}_energy_vs_substep.png")
        plt.savefig(out, dpi=120)
        plt.close(fig)

    # 2. Truncation metrics
    td = load_tag(npz, "truncation")
    if td:
        substep = td.get("substep", np.arange(len(td.get("n_truncated", []))))
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"Truncation analysis  [{label}]")

        for ax, (key, title) in zip(
            axes.ravel(),
            [
                ("n_truncated", "# Truncated vertices"),
                ("tangential_fraction", "R_tangent / (R_normal + R_tangent)"),
                ("delta_kinetic_energy", "ΔKE from truncation (J)"),
                ("delta_total_energy", "ΔE_total from truncation (J)"),
            ],
        ):
            arr = td.get(key)
            if arr is not None:
                ax.plot(substep[: len(arr)], arr)
                ax.set_xlabel("substep")
                ax.set_title(title)
                ax.grid(True, alpha=0.4)
        plt.tight_layout()
        out = os.path.join(output_dir, f"{label}_truncation.png")
        plt.savefig(out, dpi=120)
        plt.close(fig)

    # 3. Cumulative truncation energy loss
    td = load_tag(npz, "truncation")
    if td:
        dke = td.get("delta_kinetic_energy")
        dte = td.get("delta_total_energy")
        if dke is not None or dte is not None:
            substep = td.get("substep", np.arange(len(dke if dke is not None else dte)))
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.set_title(f"Cumulative truncation energy loss  [{label}]")
            if dke is not None:
                ax.plot(substep[: len(dke)], np.cumsum(dke), label="ΔKE cumul.")
            if dte is not None:
                ax.plot(substep[: len(dte)], np.cumsum(dte), label="ΔE_total cumul.", linestyle="--")
            ax.set_xlabel("substep")
            ax.set_ylabel("J (cumulative)")
            ax.legend()
            ax.grid(True, alpha=0.4)
            plt.tight_layout()
            out = os.path.join(output_dir, f"{label}_cumulative_trunc_loss.png")
            plt.savefig(out, dpi=120)
            plt.close(fig)

    # 4. Residual vs iteration (if per-iteration data present)
    it = load_tag(npz, "iterations")
    if it:
        res_mean = it.get("residual_mean")
        iters = it.get("iter")
        steps = it.get("step")
        if res_mean is not None and iters is not None:
            # Plot residual for first few steps
            unique_steps = np.unique(steps[: len(res_mean)]) if steps is not None else np.array([0])
            max_plot_steps = min(5, len(unique_steps))
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.set_title(f"Residual vs. iteration  [{label}]")
            for s in unique_steps[:max_plot_steps]:
                mask = steps[: len(res_mean)] == s
                res_s = res_mean[: len(steps)][mask]
                ax.semilogy(np.arange(len(res_s)), res_s, label=f"step {int(s)}")
            ax.set_xlabel("VBD iteration")
            ax.set_ylabel("mean ||F||")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.4)
            plt.tight_layout()
            out = os.path.join(output_dir, f"{label}_residual_vs_iter.png")
            plt.savefig(out, dpi=120)
            plt.close(fig)


# ---------------------------------------------------------------------------
# Multi-run comparison (from metadata CSV)
# ---------------------------------------------------------------------------

def analyze_metadata(meta_csv: str, output_dir: str) -> None:
    """Read an ablation metadata CSV, analyze every run, and produce comparisons."""
    if not os.path.exists(meta_csv):
        print(f"[analyze] metadata CSV not found: {meta_csv}")
        return

    rows = []
    with open(meta_csv, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("[analyze] metadata CSV is empty")
        return

    print(f"[analyze] {len(rows)} runs in metadata")
    os.makedirs(output_dir, exist_ok=True)

    # Per-run analysis
    summaries = []
    for row in rows:
        npz_path = row.get("diag_path", "")
        label = (
            f"{row.get('scene','?')}_"
            f"iter{row.get('iterations','?')}_"
            f"{row.get('ablation','?')}"
        )
        summary = analyze_single(npz_path, os.path.join(output_dir, label), label)
        summary.update(row)
        summaries.append(summary)

    # Write summary CSV
    summary_csv = os.path.join(output_dir, "summary.csv")
    all_keys = sorted({k for s in summaries for k in s})
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    print(f"[analyze] Summary CSV → {summary_csv}")

    # Comparison plots: KE ratio vs iterations for each scene/ablation
    _comparison_plots(summaries, output_dir)

    # Automated report
    _print_report(summaries)


def _comparison_plots(summaries: list[dict], output_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    scenes = sorted({s.get("scene", "") for s in summaries})
    ablations = sorted({s.get("ablation", "") for s in summaries})

    for scene in scenes:
        # KE ratio vs iteration count
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title(f"KE retention ratio vs. iterations  [{scene}]")
        for abl in ablations:
            sub = [s for s in summaries
                   if s.get("scene") == scene and s.get("ablation") == abl
                   and s.get("status") == "ok"]
            if not sub:
                continue
            try:
                iters = [int(s["iterations"]) for s in sub]
                ke_ratio = [float(s.get("end_of_step/ke_ratio", float("nan"))) for s in sub]
                order = np.argsort(iters)
                ax.plot(np.array(iters)[order], np.array(ke_ratio)[order],
                        marker="o", label=abl)
            except Exception:
                pass
        ax.set_xlabel("VBD iterations")
        ax.set_ylabel("KE_final / KE_initial")
        ax.set_xscale("log")
        ax.legend()
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        out = os.path.join(output_dir, f"{scene}_ke_ratio_vs_iters.png")
        plt.savefig(out, dpi=120)
        plt.close(fig)

        # Tangential fraction vs iteration count
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title(f"Tangential truncation fraction vs. iterations  [{scene}]")
        for abl in ablations:
            sub = [s for s in summaries
                   if s.get("scene") == scene and s.get("ablation") == abl
                   and s.get("status") == "ok"]
            if not sub:
                continue
            try:
                iters = [int(s["iterations"]) for s in sub]
                tf = [float(s.get("truncation/mean_tangential_fraction", float("nan"))) for s in sub]
                order = np.argsort(iters)
                ax.plot(np.array(iters)[order], np.array(tf)[order],
                        marker="s", label=abl)
            except Exception:
                pass
        ax.set_xlabel("VBD iterations")
        ax.set_ylabel("mean R_tangent / (R_n + R_t)")
        ax.set_xscale("log")
        ax.legend()
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        out = os.path.join(output_dir, f"{scene}_tangential_fraction_vs_iters.png")
        plt.savefig(out, dpi=120)
        plt.close(fig)


def _print_report(summaries: list[dict]) -> None:
    """Print automated interpretation of damping sources."""
    print("\n" + "=" * 60)
    print("  DAMPING SOURCE INTERPRETATION")
    print("=" * 60)

    scenes = sorted({s.get("scene", "") for s in summaries})
    for scene in scenes:
        print(f"\n[Scene: {scene}]")
        full = [s for s in summaries if s.get("scene") == scene and s.get("ablation") == "full"]
        no_contact = [s for s in summaries if s.get("scene") == scene and s.get("ablation") == "no_contact"]
        no_trunc = [s for s in summaries if s.get("scene") == scene and s.get("ablation") == "no_trunc"]

        # Check if damping decreases with more iterations → under-convergence
        if full:
            try:
                iters_ke = sorted(
                    [(int(s["iterations"]), float(s.get("end_of_step/ke_ratio", float("nan"))))
                     for s in full],
                    key=lambda x: x[0]
                )
                if len(iters_ke) >= 2:
                    ke_low = iters_ke[0][1]
                    ke_high = iters_ke[-1][1]
                    if ke_high > ke_low + 0.02:
                        print(f"  ✓ KE retention improves with iterations "
                              f"({ke_low:.3f}→{ke_high:.3f}): "
                              "likely UNDER-CONVERGENCE / slow coupling propagation.")
                    else:
                        print(f"  – KE retention flat across iterations ({ke_low:.3f}→{ke_high:.3f}).")
            except Exception:
                pass

        # Check truncation energy loss
        trunc_loss_vals = [
            float(s.get("truncation/cumulative_delta_total_energy", float("nan")))
            for s in full
        ]
        trunc_loss_vals = [v for v in trunc_loss_vals if not np.isnan(v)]
        if trunc_loss_vals:
            avg_loss = np.mean(trunc_loss_vals)
            if avg_loss < -1e-4:
                print(f"  ✓ Cumulative truncation ΔE_total = {avg_loss:.4e} J: "
                      "likely TRUNCATION / SAFETY-FILTER DAMPING.")

        # Check tangential fraction
        tf_vals = [
            float(s.get("truncation/overall_tangential_fraction", float("nan")))
            for s in full
        ]
        tf_vals = [v for v in tf_vals if not np.isnan(v)]
        if tf_vals:
            avg_tf = np.mean(tf_vals)
            if avg_tf > 0.3:
                print(f"  ✓ Mean tangential fraction = {avg_tf:.2f}: "
                      "likely TANGENTIAL CLIPPING damping.")
            else:
                print(f"  – Tangential fraction = {avg_tf:.2f} (mostly normal removal).")

        # Check if contact=off still damps
        if full and no_contact:
            try:
                ke_full = np.mean([float(s.get("end_of_step/ke_ratio", float("nan"))) for s in full])
                ke_nc = np.mean([float(s.get("end_of_step/ke_ratio", float("nan"))) for s in no_contact])
                if not np.isnan(ke_full) and not np.isnan(ke_nc):
                    if ke_nc < ke_full - 0.02:
                        print(f"  – Contact OFF still damps (ke_ratio {ke_nc:.3f} < {ke_full:.3f}): "
                              "IMPLICIT SOLVER damping present even without contact.")
                    else:
                        print(f"  ✓ Disabling contact reduces damping: contact is a primary cause.")
            except Exception:
                pass

    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze VBD artificial-damping diagnostic logs."
    )
    parser.add_argument("--metadata-csv", default=None,
                        help="Path to metadata.csv from run_damping_ablation.py.")
    parser.add_argument("--npz", default=None,
                        help="Single diagnostics.npz file to analyze.")
    parser.add_argument("--output-dir", default="analysis",
                        help="Directory for plots and summary CSV.")
    args = parser.parse_args()

    if args.npz:
        label = os.path.basename(os.path.dirname(args.npz)) or "run"
        summary = analyze_single(args.npz, args.output_dir, label)
        print("\n--- Single-run summary ---")
        for k, v in sorted(summary.items()):
            if k not in ("path", "status"):
                print(f"  {k}: {v}")
        print(f"\n  Plots → {args.output_dir}/")

    if args.metadata_csv:
        analyze_metadata(args.metadata_csv, args.output_dir)

    if not args.npz and not args.metadata_csv:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
