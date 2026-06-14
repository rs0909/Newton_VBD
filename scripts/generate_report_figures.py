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
"""Generate time-series figures for the VBD damping ablation report.

Reads the existing per-run ``diagnostics.npz`` logs produced by
``scripts/run_damping_ablation.py`` (already present under ``runs/``) and
renders a curated set of figures referenced from
``docs/research_notes/vbd_damping_ablation_report.md``.

This script does not run any new simulations — it only reads existing
``.npz`` logs and writes PNG figures to ``docs/research_notes/figures/``.

Usage:
    uv run scripts/generate_report_figures.py
"""

from __future__ import annotations

import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(_REPO_ROOT, "runs")
OUT_DIR = os.path.join(_REPO_ROOT, "docs", "research_notes", "figures")

# Consistent colors across figures.
ITER_COLORS = {10: "tab:blue", 50: "tab:orange", 100: "tab:green", 500: "tab:red"}
ABL_COLORS = {
    "full": "tab:blue",
    "no_contact": "tab:orange",
    "no_trunc": "tab:green",
    "no_friction": "tab:red",
}
SUB_COLORS = {2: "tab:purple", 5: "tab:brown", 10: "tab:green", 20: "tab:orange", 40: "tab:blue"}


def _load(run_dir: str) -> np.lib.npyio.NpzFile:
    return np.load(os.path.join(RUNS_DIR, run_dir, "diagnostics.npz"))


def _time(npz: np.lib.npyio.NpzFile) -> np.ndarray:
    """Cumulative simulated time (s) at each logged checkpoint.

    All per-substep tags (``end_of_step``, ``truncation``, ...) share the
    same step/substep index, so ``end_of_step/dt`` is used as the timestep
    sequence for every tag.
    """
    return np.cumsum(npz["end_of_step/dt"])


def _savefig(fig, name: str) -> None:
    path = os.path.join(OUT_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, _REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Scene A — no_contact_oscillation
# ---------------------------------------------------------------------------

def fig01_sceneA_energy_decay() -> None:
    """Total energy vs time for Scene A, full variant, across iteration counts.

    Reproduces report §3.1/§3.3 finding 2: iteration count has no effect on
    the energy decay curve.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for it in (10, 50, 100, 500):
        npz = _load(f"ablation_sceneA/no_contact_oscillation_iter{it}_dtm1.0000_full")
        ax.plot(_time(npz), npz["end_of_step/total_energy"], label=f"iter={it}",
                color=ITER_COLORS[it], alpha=0.8)
    ax.set_xlabel("simulated time (s)")
    ax.set_ylabel("total energy (J)")
    ax.set_title("Scene A — Total Energy vs Time (full, varying iterations)")
    ax.legend()
    ax.grid(alpha=0.3)
    _savefig(fig, "fig01_sceneA_energy_decay.png")


def fig02_sceneA_energy_components() -> None:
    """Energy component breakdown for Scene A, iter=100, full.

    Supports report §3.2 (per-substep energy budget): shows where the
    initial energy goes (kinetic -> dissipated, gravitational PE released,
    stretch/bending small for this scene).
    """
    npz = _load("ablation_sceneA/no_contact_oscillation_iter100_dtm1.0000_full")
    t = _time(npz)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, npz["end_of_step/total_energy"], label="total", color="black", lw=2)
    ax.plot(t, npz["end_of_step/kinetic_energy"], label="kinetic")
    ax.plot(t, npz["end_of_step/gravitational_potential"], label="gravitational PE")
    ax.plot(t, npz["end_of_step/stretch_energy"], label="stretch")
    ax.plot(t, npz["end_of_step/bending_energy"], label="bending")
    ax.set_xlabel("simulated time (s)")
    ax.set_ylabel("energy (J)")
    ax.set_title("Scene A — Energy Components vs Time (iter=100, full)")
    ax.legend()
    ax.grid(alpha=0.3)
    _savefig(fig, "fig02_sceneA_energy_components.png")


# ---------------------------------------------------------------------------
# Scene B — frictionless_sliding
# ---------------------------------------------------------------------------

def fig03_sceneB_energy_by_ablation() -> None:
    """Total energy vs time for Scene B, iter=100, across ablation variants.

    Supports report §4.1: shows the `no_contact` variant diverging (free
    fall through the bottom layer) while `full`/`no_trunc`/`no_friction`
    settle to similar negative total-energy values.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for abl in ("full", "no_contact", "no_trunc", "no_friction"):
        npz = _load(f"ablation_sceneB/frictionless_sliding_iter100_dtm1.0000_{abl}")
        ax.plot(_time(npz), npz["end_of_step/total_energy"], label=abl, color=ABL_COLORS[abl])
    ax.set_xlabel("simulated time (s)")
    ax.set_ylabel("total energy (J)")
    ax.set_title("Scene B — Total Energy vs Time (iter=100, by ablation)")
    ax.legend()
    ax.grid(alpha=0.3)
    _savefig(fig, "fig03_sceneB_energy_by_ablation.png")


def fig04_sceneB_truncation_vs_iter() -> None:
    """Planar-DAT truncation fraction and tangential fraction vs time, Scene B (full).

    Supports report §4.2/§4.3: more iterations -> larger truncated
    fraction, while tangential fraction stays approximately constant
    (~0.41-0.43) across iteration counts.
    """
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for it in (10, 50, 100, 500):
        npz = _load(f"ablation_sceneB/frictionless_sliding_iter{it}_dtm1.0000_full")
        t = _time(npz)
        axes[0].plot(t, npz["truncation/fraction_truncated"], label=f"iter={it}",
                      color=ITER_COLORS[it], alpha=0.8)
        axes[1].plot(t, npz["truncation/tangential_fraction"], label=f"iter={it}",
                      color=ITER_COLORS[it], alpha=0.8)
    axes[0].set_ylabel("fraction truncated")
    axes[0].set_title("Scene B — Planar-DAT Truncation vs Iteration Count (full)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].set_ylabel("tangential fraction")
    axes[1].set_xlabel("simulated time (s)")
    axes[1].grid(alpha=0.3)
    _savefig(fig, "fig04_sceneB_truncation_vs_iter.png")


# ---------------------------------------------------------------------------
# Scene C — separating_contact
# ---------------------------------------------------------------------------

def fig05_sceneC_energy_by_iter() -> None:
    """Total energy vs time for Scene C, full variant, across iteration counts.

    Supports report §5.1/§5.2: shows the convergence threshold at iter~50
    and the iter=10 divergence beginning around frame 17 (t~0.28s).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for it in (10, 50, 100, 500):
        npz = _load(f"ablation_sceneC/separating_contact_iter{it}_dtm1.0000_full")
        ax.plot(_time(npz), npz["end_of_step/total_energy"], label=f"iter={it}",
                color=ITER_COLORS[it], alpha=0.8)
    ax.set_xlabel("simulated time (s)")
    ax.set_ylabel("total energy (J)")
    ax.set_title("Scene C — Total Energy vs Time (full, varying iterations)")
    ax.legend()
    ax.grid(alpha=0.3)
    _savefig(fig, "fig05_sceneC_energy_by_iter.png")


def fig06_sceneC_iter10_stability() -> None:
    """Total energy vs time for Scene C, iter=10, full vs no_trunc vs no_contact.

    Supports report §5.2/§5.3: `no_trunc` at iter=10 goes unstable
    (negative total energy) while `full` and `no_contact` remain bounded —
    Planar-DAT acts as a stabilizer at low iteration counts.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for abl in ("full", "no_trunc", "no_contact"):
        npz = _load(f"ablation_sceneC/separating_contact_iter10_dtm1.0000_{abl}")
        ax.plot(_time(npz), npz["end_of_step/total_energy"], label=abl, color=ABL_COLORS[abl])
    ax.axhline(0.0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("simulated time (s)")
    ax.set_ylabel("total energy (J)")
    ax.set_title("Scene C — iter=10 Stability: full vs no_trunc vs no_contact")
    ax.legend()
    ax.grid(alpha=0.3)
    _savefig(fig, "fig06_sceneC_iter10_stability.png")


# ---------------------------------------------------------------------------
# Scene D — twist_release (720-frame, 12s)
# ---------------------------------------------------------------------------

def fig07_sceneD_energy_timeline() -> None:
    """Total energy and contact-pair count vs time, Scene D, iter=100.

    Supports report §6.1: shows the twist-then-release timeline for `full`
    vs `no_contact`, with contact-pair count (full) on a secondary axis.
    """
    npz_full = _load("ablation_sceneD_720/twist_release_iter100_dtm1.0000_full")
    npz_nc = _load("ablation_sceneD_720/twist_release_iter100_dtm1.0000_no_contact")
    t_full = _time(npz_full)
    t_nc = _time(npz_nc)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t_full, npz_full["end_of_step/total_energy"], label="total energy (full)",
            color=ABL_COLORS["full"])
    ax.plot(t_nc, npz_nc["end_of_step/total_energy"], label="total energy (no_contact)",
            color=ABL_COLORS["no_contact"])
    ax.axvline(10.0, color="gray", lw=1.0, ls="--", label="release (t=10s)")
    ax.set_xlabel("simulated time (s)")
    ax.set_ylabel("total energy (J)")

    ax2 = ax.twinx()
    ax2.plot(t_full, npz_full["end_of_step/n_contacts"], label="n_contacts (full)",
             color="tab:gray", alpha=0.6, lw=1.0)
    ax2.set_ylabel("contact pair count")

    ax.set_title("Scene D — Energy Timeline (iter=100): twist (0-10s) then release")
    fig.legend(*ax.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.08, 0.95))
    ax2.legend(loc="upper right")
    ax.grid(alpha=0.3)
    _savefig(fig, "fig07_sceneD_energy_timeline.png")


def fig08_sceneD_final_energy_by_iter() -> None:
    """Grouped bar chart of final (t=12s) total energy by iteration count and ablation.

    Reproduces report §6.2 table.
    """
    iters = (10, 50, 100, 500)
    ablations = ("full", "no_contact", "no_trunc", "no_friction")
    values = {abl: [] for abl in ablations}
    for it in iters:
        for abl in ablations:
            npz = _load(f"ablation_sceneD_720/twist_release_iter{it}_dtm1.0000_{abl}")
            values[abl].append(npz["end_of_step/total_energy"][-1])

    x = np.arange(len(iters))
    width = 0.2
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, abl in enumerate(ablations):
        ax.bar(x + (i - 1.5) * width, values[abl], width, label=abl, color=ABL_COLORS[abl])
    ax.set_xticks(x)
    ax.set_xticklabels([str(it) for it in iters])
    ax.set_xlabel("iterations")
    ax.set_ylabel("final total energy at t=12s (J)")
    ax.set_title("Scene D — Final Energy (t=12s) by Iteration Count and Ablation")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    _savefig(fig, "fig08_sceneD_final_energy_by_iter.png")


def fig09_sceneD_truncation_stats() -> None:
    """Truncation pair count and tangential fraction vs time, Scene D (full).

    Supports report §6.3/finding 2: per-substep truncation magnitude is
    tiny (n_truncated ~30-100 out of ~40,000 contacts) regardless of
    iteration count, while tangential fraction stays ~0.4.
    """
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for it in (10, 50, 100, 500):
        npz = _load(f"ablation_sceneD_720/twist_release_iter{it}_dtm1.0000_full")
        t = _time(npz)
        axes[0].plot(t, npz["truncation/n_truncated"], label=f"iter={it}",
                      color=ITER_COLORS[it], alpha=0.8)
        axes[1].plot(t, npz["truncation/tangential_fraction"], label=f"iter={it}",
                      color=ITER_COLORS[it], alpha=0.8)
    axes[0].axvline(10.0, color="gray", lw=1.0, ls="--")
    axes[1].axvline(10.0, color="gray", lw=1.0, ls="--")
    axes[0].set_ylabel("n_truncated")
    axes[0].set_title("Scene D — Planar-DAT Truncation Statistics (full)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].set_ylabel("tangential fraction")
    axes[1].set_xlabel("simulated time (s)")
    axes[1].grid(alpha=0.3)
    _savefig(fig, "fig09_sceneD_truncation_stats.png")


# ---------------------------------------------------------------------------
# Cross-scene synthesis
# ---------------------------------------------------------------------------

def fig10_cross_scene_planar_dat_isolation() -> None:
    """Grouped bar chart of (full - no_trunc) final total energy per scene/iter.

    Reproduces report §7.2 table (Planar-DAT isolation).
    """
    iters = (10, 50, 100, 500)
    scenes = {
        "A": ("ablation_sceneA", "no_contact_oscillation"),
        "B": ("ablation_sceneB", "frictionless_sliding"),
        "C": ("ablation_sceneC", "separating_contact"),
        "D": ("ablation_sceneD_720", "twist_release"),
    }

    diffs = {scene_id: [] for scene_id in scenes}
    for scene_id, (run_dir, scene_name) in scenes.items():
        for it in iters:
            npz_full = _load(f"{run_dir}/{scene_name}_iter{it}_dtm1.0000_full")
            npz_notrunc = _load(f"{run_dir}/{scene_name}_iter{it}_dtm1.0000_no_trunc")
            diff = npz_full["end_of_step/total_energy"][-1] - npz_notrunc["end_of_step/total_energy"][-1]
            diffs[scene_id].append(diff)

    x = np.arange(len(scenes))
    width = 0.2
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, it in enumerate(iters):
        vals = [diffs[scene_id][i] for scene_id in scenes]
        ax.bar(x + (i - 1.5) * width, vals, width, label=f"iter={it}", color=ITER_COLORS[it])
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(list(scenes.keys()))
    ax.set_xlabel("scene")
    ax.set_ylabel("E_final(full) - E_final(no_trunc)  (J)")
    ax.set_title("Cross-Scene Planar-DAT Isolation (full - no_trunc)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    _savefig(fig, "fig10_cross_scene_planar_dat_isolation.png")


# ---------------------------------------------------------------------------
# Substep sweep (§7.5)
# ---------------------------------------------------------------------------

def fig11_substep_sceneA_energy() -> None:
    """Total energy vs physical time for Scene A, full, varying substep count.

    Supports report §7.5 Scene A: structural implicit damping depends on
    dt (fewer substeps -> larger dt -> more energy lost over the same
    0.5s simulated-time window).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for sub in (2, 5, 10, 20, 40):
        npz = _load(f"ablation_substep/no_contact_oscillation_iter100_sub{sub}_dtm1.0000_full")
        ax.plot(_time(npz), npz["end_of_step/total_energy"], label=f"substeps={sub}",
                color=SUB_COLORS[sub], alpha=0.8)
    ax.axhline(0.0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("simulated time (s)")
    ax.set_ylabel("total energy (J)")
    ax.set_title("Scene A — Total Energy vs Time by Substep Count (full, iter=100)")
    ax.legend()
    ax.grid(alpha=0.3)
    _savefig(fig, "fig11_substep_sceneA_energy.png")


def fig12_substep_sceneB_truncation() -> None:
    """Truncated-pair count vs physical time for Scene B, full, varying substep count.

    Supports report §7.5 Scene B: Planar-DAT truncation rate vs dt is
    non-monotonic (burst-mode truncation at intermediate dt).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for sub in (2, 5, 10, 20, 40):
        npz = _load(f"ablation_substep/frictionless_sliding_iter100_sub{sub}_dtm1.0000_full")
        t = _time(npz)
        ax.plot(t, npz["truncation/n_truncated"], label=f"substeps={sub}",
                color=SUB_COLORS[sub], alpha=0.8)
    ax.set_xlabel("simulated time (s)")
    ax.set_ylabel("n_truncated")
    ax.set_title("Scene B — Truncated Pair Count vs Time by Substep Count (full, iter=100)")
    ax.legend()
    ax.grid(alpha=0.3)
    _savefig(fig, "fig12_substep_sceneB_truncation.png")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    figures = [
        fig01_sceneA_energy_decay,
        fig02_sceneA_energy_components,
        fig03_sceneB_energy_by_ablation,
        fig04_sceneB_truncation_vs_iter,
        fig05_sceneC_energy_by_iter,
        fig06_sceneC_iter10_stability,
        fig07_sceneD_energy_timeline,
        fig08_sceneD_final_energy_by_iter,
        fig09_sceneD_truncation_stats,
        fig10_cross_scene_planar_dat_isolation,
        fig11_substep_sceneA_energy,
        fig12_substep_sceneB_truncation,
    ]
    print(f"[figures] writing {len(figures)} figures to {os.path.relpath(OUT_DIR, _REPO_ROOT)}/")
    for fn in figures:
        print(f"[figures] {fn.__name__}")
        fn()
    print("[figures] done.")


if __name__ == "__main__":
    main()
