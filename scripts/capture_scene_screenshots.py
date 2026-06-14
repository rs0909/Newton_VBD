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
"""Capture representative scene screenshots for the VBD damping ablation report.

Re-runs each of the four benchmark scenes from
``newton.examples.cloth.example_cloth_diagnostics`` with the same
configuration as the `full` ablation variant (iterations=100, substeps=10,
dt=frame_dt/substeps), without diagnostics logging, and renders the cloth
mesh at a few representative frames using matplotlib's mplot3d
(``Poly3DCollection``). No GL/pyglet/viewer is used.

This script does not modify any existing run outputs — it only writes new
PNGs to ``docs/research_notes/figures/scenes/``.

Usage:
    uv run scripts/capture_scene_screenshots.py
"""

from __future__ import annotations

import importlib
import os
import sys
import types

import numpy as np
import warp as wp

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

OUT_DIR = os.path.join(_REPO_ROOT, "docs", "research_notes", "figures", "scenes")

# (elevation, azimuth) for each scene's 3D view.
CAMERA = {
    "A": (20, -60),
    "B": (20, -60),
    "C": (20, -60),
    "D": (25, -60),
}


def _make_args(scene: str) -> types.SimpleNamespace:
    """Same configuration as the `full` ablation variant (iter=100, full ablation)."""
    a = types.SimpleNamespace()
    a.scene = scene
    a.iterations = 100
    a.dt = -1.0
    a.substeps = 10
    a.no_contact = False
    a.no_truncation = False
    a.no_friction = False
    a.no_gravity = False
    a.stiffness_scale = 1.0
    a._diag = None
    return a


def _run_and_capture(mod, build_fn, args, n_frames: int, capture_frames: set[int]):
    """Run the scene and return {frame_num: particle_q (N,3)} plus tri_indices (T,3)."""
    import newton
    import newton.examples

    scene = build_fn(args)
    model = scene["model"]
    solver = scene["solver"]
    twist_data = scene.get("twist_data")

    if "state_0" in scene:
        state_0, state_1 = scene["state_0"], scene["state_1"]
    else:
        state_0, state_1 = model.state(), model.state()

    control = model.control()
    pipeline = newton.examples.create_collision_pipeline(model, None)
    tri_indices = model.tri_indices.numpy()

    fps = 60
    frame_dt = 1.0 / fps
    sim_dt = args.dt if args.dt > 0 else frame_dt / args.substeps

    captured = {}
    if 0 in capture_frames:
        captured[0] = state_0.particle_q.numpy().copy()

    for frame in range(n_frames):
        contacts = model.collide(state_0, collision_pipeline=pipeline)
        for _ in range(args.substeps):
            state_0.clear_forces()
            if twist_data is not None:
                wp.launch(
                    kernel=mod._apply_twist_rotation,
                    dim=twist_data["rot_indices"].shape[0],
                    inputs=[
                        twist_data["rot_indices"],
                        twist_data["rot_axes"],
                        twist_data["roots"],
                        twist_data["roots_to_ps"],
                        twist_data["t"],
                        twist_data["angular_velocity"],
                        sim_dt,
                        twist_data["end_time"],
                    ],
                    outputs=[state_0.particle_q, state_1.particle_q],
                )
            solver.step(state_0, state_1, control, contacts, sim_dt)
            state_0, state_1 = state_1, state_0

        frame_num = frame + 1
        if frame_num in capture_frames:
            captured[frame_num] = state_0.particle_q.numpy().copy()

    return captured, tri_indices


def _set_equal_axes(ax, pts: np.ndarray, pad: float = 0.1) -> None:
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = (maxs - mins).max() / 2.0 * (1.0 + pad)
    radius = max(radius, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def _add_mesh(ax, q: np.ndarray, tris: np.ndarray, facecolor: str) -> None:
    if len(tris) == 0:
        return
    verts = q[tris]
    poly = Poly3DCollection(verts, facecolor=facecolor, edgecolor="k", linewidths=0.1, alpha=0.95)
    ax.add_collection3d(poly)


def _render_scene(out_name: str, title: str, frames: dict, tri_indices: np.ndarray,
                   labels: dict, camera: tuple[float, float],
                   layer_masks: tuple[np.ndarray, np.ndarray] | None = None) -> None:
    all_pts = np.concatenate(list(frames.values()), axis=0)
    frame_nums = sorted(frames.keys())

    fig = plt.figure(figsize=(13, 4.5))
    for i, frame_num in enumerate(frame_nums):
        q = frames[frame_num]
        ax = fig.add_subplot(1, len(frame_nums), i + 1, projection="3d")
        if layer_masks is None:
            _add_mesh(ax, q, tri_indices, facecolor="lightsteelblue")
        else:
            mask_top, mask_bottom = layer_masks
            _add_mesh(ax, q, tri_indices[mask_bottom], facecolor="salmon")
            _add_mesh(ax, q, tri_indices[mask_top], facecolor="lightsteelblue")
        _set_equal_axes(ax, all_pts)
        ax.view_init(elev=camera[0], azim=camera[1])
        ax.set_title(labels[frame_num], fontsize=10)
        ax.set_xlabel("x", fontsize=8)
        ax.set_ylabel("y", fontsize=8)
        ax.set_zlabel("z", fontsize=8)
        ax.tick_params(labelsize=6)

    fig.suptitle(title)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, out_name)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, _REPO_ROOT)}")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    wp.set_device("cuda:0")

    mod = importlib.import_module("newton.examples.cloth.example_cloth_diagnostics")

    # --- Scene A: no_contact_oscillation (30 frames = 0.5s) ---
    print("[screenshots] Scene A — no_contact_oscillation")
    args_a = _make_args("no_contact_oscillation")
    frames_a, tris_a = _run_and_capture(
        mod, mod.build_scene_no_contact_oscillation, args_a,
        n_frames=29, capture_frames={0, 15, 29},
    )
    _render_scene(
        "scene_A_no_contact_oscillation.png",
        "Scene A — no_contact_oscillation (full, iter=100)",
        frames_a, tris_a,
        labels={0: "t=0.00s (initial)", 15: "t=0.25s", 29: "t=0.48s (final)"},
        camera=CAMERA["A"],
    )

    # --- Scene B: frictionless_sliding (30 frames = 0.5s) ---
    print("[screenshots] Scene B — frictionless_sliding")
    args_b = _make_args("frictionless_sliding")
    frames_b, tris_b = _run_and_capture(
        mod, mod.build_scene_frictionless_sliding, args_b,
        n_frames=29, capture_frames={0, 15, 29},
    )
    # Layer 1 (top, sliding) uses particle indices [0, 625); layer 2 (bottom, pinned) >= 625.
    n_layer1 = 25 * 25
    mask_top = np.all(tris_b < n_layer1, axis=1)
    mask_bottom = np.all(tris_b >= n_layer1, axis=1)
    _render_scene(
        "scene_B_frictionless_sliding.png",
        "Scene B — frictionless_sliding (full, iter=100); blue=top layer, red=bottom layer",
        frames_b, tris_b,
        labels={0: "t=0.00s (initial)", 15: "t=0.25s", 29: "t=0.48s (final)"},
        camera=CAMERA["B"],
        layer_masks=(mask_top, mask_bottom),
    )

    # --- Scene C: separating_contact (30 frames = 0.5s) ---
    print("[screenshots] Scene C — separating_contact")
    args_c = _make_args("separating_contact")
    frames_c, tris_c = _run_and_capture(
        mod, mod.build_scene_separating_contact, args_c,
        n_frames=29, capture_frames={0, 18, 29},
    )
    _render_scene(
        "scene_C_separating_contact.png",
        "Scene C — separating_contact (full, iter=100)",
        frames_c, tris_c,
        labels={0: "t=0.00s (initial)", 18: "t=0.30s (contact)", 29: "t=0.48s (final)"},
        camera=CAMERA["C"],
    )

    # --- Scene D: twist_release (720 frames = 12s) ---
    print("[screenshots] Scene D — twist_release (720 frames, this takes a few minutes)")
    args_d = _make_args("twist_release")
    frames_d, tris_d = _run_and_capture(
        mod, mod.build_scene_twist_release, args_d,
        n_frames=720, capture_frames={0, 480, 720},
    )
    _render_scene(
        "scene_D_twist_release.png",
        "Scene D — twist_release (full, iter=100)",
        frames_d, tris_d,
        labels={0: "t=0s (rest)", 480: "t=8s (peak contact)", 720: "t=12s (after release)"},
        camera=CAMERA["D"],
    )

    print("[screenshots] done.")


if __name__ == "__main__":
    main()
