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
"""Ablation sweep for VBD artificial-damping analysis.

Runs the four benchmark scenes across multiple configurations (solver
iteration count, collision mode, timestep) and saves per-run diagnostic
logs plus a metadata CSV.

Usage:
    uv run scripts/run_damping_ablation.py --output-dir runs/damping_ablation
    uv run scripts/run_damping_ablation.py --quick   # fast smoke-test (2 frames)
    uv run scripts/run_damping_ablation.py --scene no_contact_oscillation --iterations 10 50 100

Produces:
    <output-dir>/
        metadata.csv                      # one row per run
        <scene>_<config>/diagnostics.npz  # per-run VBDDiagnostics log
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import sys
import time
import traceback
import types

# Add repo root to path so we can import newton without uv run
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _make_args(
    scene: str,
    iterations: int,
    dt: float,
    substeps: int,
    no_contact: bool,
    no_truncation: bool,
    no_friction: bool,
    diag_output_path: str,
    diag_log_iterations: bool = False,
    stiffness_scale: float = 1.0,
    device: str = "cuda:0",
) -> types.SimpleNamespace:
    a = types.SimpleNamespace()
    a.scene = scene
    a.diagnostics = True
    a.diag_output_path = diag_output_path
    a.diag_log_iterations = diag_log_iterations
    a.diag_log_frequency = 1
    a.iterations = iterations
    a.dt = dt
    a.substeps = substeps
    a.no_contact = no_contact
    a.no_truncation = no_truncation
    a.no_friction = no_friction
    a.no_gravity = False
    a.stiffness_scale = stiffness_scale
    a.device = device
    a._diag = None
    return a


def _run_one(run_args: types.SimpleNamespace, n_frames: int, output_dir: str) -> dict:
    """Execute a single simulation run and return metadata."""
    import numpy as np
    import warp as wp

    wp.set_device(getattr(run_args, "device", "cuda:0"))

    # Import the diagnostics benchmark module
    mod = importlib.import_module("newton.examples.cloth.example_cloth_diagnostics")

    diag = mod.VBDDiagnostics(
        output_path=run_args.diag_output_path,
        log_iterations=run_args.diag_log_iterations,
        log_frequency=run_args.diag_log_frequency,
        log_truncation=True,
    )
    run_args._diag = diag

    # Build scene
    scene_name = run_args.scene
    builders = {
        "no_contact_oscillation": mod.build_scene_no_contact_oscillation,
        "frictionless_sliding": mod.build_scene_frictionless_sliding,
        "separating_contact": mod.build_scene_separating_contact,
        "twist_release": mod.build_scene_twist_release,
    }
    build_fn = builders[scene_name]

    t0 = time.perf_counter()
    try:
        scene = build_fn(run_args)
    except Exception as e:
        return {"status": "build_error", "error": str(e)}

    model = scene["model"]
    solver = scene["solver"]
    twist_data = scene.get("twist_data")

    if "state_0" in scene:
        state_0 = scene["state_0"]
        state_1 = scene["state_1"]
    else:
        state_0 = model.state()
        state_1 = model.state()

    control = model.control()

    import newton
    import newton.examples
    collision_pipeline = newton.examples.create_collision_pipeline(model, None)
    contacts = model.collide(state_0, collision_pipeline=collision_pipeline)

    fps = 60
    frame_dt = 1.0 / fps
    sim_substeps = run_args.substeps
    sim_dt = run_args.dt if run_args.dt > 0 else frame_dt / sim_substeps
    global_substep = 0

    try:
        for frame in range(n_frames):
            contacts = model.collide(state_0, collision_pipeline=collision_pipeline)

            for _ in range(sim_substeps):
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

                diag.begin_step(
                    step_num=frame,
                    substep_num=global_substep,
                    dt=sim_dt,
                )
                solver.step(state_0, state_1, control, contacts, sim_dt)
                global_substep += 1
                state_0, state_1 = state_1, state_0

    except Exception as e:
        traceback.print_exc()
        meta = {"status": "runtime_error", "error": str(e)}
        diag.save()
        return meta

    diag.save()
    elapsed = time.perf_counter() - t0

    # Compute a quick summary from the saved log
    summary = {}
    try:
        npz = np.load(run_args.diag_output_path)
        ke_key = "end_of_step/kinetic_energy"
        if ke_key in npz:
            ke = npz[ke_key]
            summary["ke_initial"] = float(ke[0]) if len(ke) > 0 else float("nan")
            summary["ke_final"] = float(ke[-1]) if len(ke) > 0 else float("nan")
            summary["ke_ratio"] = summary["ke_final"] / max(summary["ke_initial"], 1e-30)
        trunc_key = "truncation/n_truncated"
        if trunc_key in npz:
            nt = npz[trunc_key]
            summary["mean_n_truncated"] = float(nt.mean()) if len(nt) > 0 else 0.0
        tang_key = "truncation/tangential_fraction"
        if tang_key in npz:
            tf = npz[tang_key]
            summary["mean_tangential_fraction"] = float(tf.mean()) if len(tf) > 0 else 0.0
    except Exception:
        pass

    return {
        "status": "ok",
        "wall_time_s": elapsed,
        "n_frames": n_frames,
        "n_substeps": global_substep,
        **summary,
    }


def main():
    parser = argparse.ArgumentParser(
        description="VBD artificial-damping ablation sweep."
    )
    parser.add_argument("--output-dir", default="runs/damping_ablation",
                        help="Root directory for all run outputs.")
    parser.add_argument("--n-frames", type=int, default=30,
                        help="Number of rendered frames per run.")
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke-test: 2 frames, small sweep.")
    parser.add_argument(
        "--scene", nargs="+",
        choices=["no_contact_oscillation", "frictionless_sliding",
                 "separating_contact", "twist_release"],
        default=None,
        help="Scenes to run (default: all four).",
    )
    parser.add_argument("--iterations", nargs="+", type=int, default=None,
                        help="VBD iteration counts to sweep (e.g. 10 50 100 500).")
    parser.add_argument("--dt", nargs="+", type=float, default=None,
                        help="Timestep values in seconds (e.g. 0.001 0.002).")
    parser.add_argument("--substeps", nargs="+", type=int, default=None,
                        help="Substeps per frame — accepts multiple values for sweep (e.g. 2 5 10 20 40).")
    parser.add_argument("--log-iterations", action="store_true",
                        help="Enable per-iteration logging (very slow).")
    parser.add_argument("--device", default="cuda:0",
                        help="Warp device to use (e.g. cuda:0, cuda:1, cpu).")
    args = parser.parse_args()

    if args.quick:
        scenes = ["no_contact_oscillation"]
        iter_counts = [10, 50]
        dt_values = [-1.0]
        substep_values = [10]
        n_frames = 2
    else:
        scenes = args.scene or ["no_contact_oscillation", "frictionless_sliding",
                                 "separating_contact"]
        iter_counts = args.iterations or [10, 50, 100, 500]
        dt_values = args.dt or [-1.0]
        substep_values = args.substeps or [10]
        n_frames = args.n_frames

    # Ablation matrix: [contact on/off, truncation on/off]
    ablation_flags = [
        ("full",        False, False, False),   # (label, no_contact, no_truncation, no_friction)
        ("no_contact",  True,  False, False),
        ("no_trunc",    False, True,  False),
        ("no_friction", False, False, True),
    ]

    os.makedirs(args.output_dir, exist_ok=True)
    meta_path = os.path.join(args.output_dir, "metadata.csv")

    fieldnames = [
        "run_id", "scene", "iterations", "substeps", "dt", "ablation",
        "no_contact", "no_truncation", "no_friction",
        "status", "wall_time_s", "n_frames", "n_substeps",
        "ke_initial", "ke_final", "ke_ratio",
        "mean_n_truncated", "mean_tangential_fraction",
        "diag_path",
    ]

    with open(meta_path, "w", newline="") as meta_f:
        writer = csv.DictWriter(meta_f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        run_id = 0
        total = (len(scenes) * len(iter_counts) * len(dt_values)
                 * len(substep_values) * len(ablation_flags))
        print(f"[ablation] {total} runs planned → {args.output_dir}")

        for scene in scenes:
            for n_iter in iter_counts:
                for dt_val in dt_values:
                    for n_sub in substep_values:
                        for abl_label, no_contact, no_trunc, no_fric in ablation_flags:
                            run_label = (
                                f"{scene}_iter{n_iter}_sub{n_sub}_dt{dt_val:.4f}_{abl_label}"
                                .replace("-", "m")
                            )
                            diag_path = os.path.join(
                                args.output_dir, run_label, "diagnostics.npz"
                            )
                            os.makedirs(os.path.dirname(diag_path), exist_ok=True)

                            run_args = _make_args(
                                scene=scene,
                                iterations=n_iter,
                                dt=dt_val,
                                substeps=n_sub,
                                no_contact=no_contact,
                                no_truncation=no_trunc,
                                no_friction=no_fric,
                                diag_output_path=diag_path,
                                diag_log_iterations=args.log_iterations,
                                device=args.device,
                            )

                            print(f"  [{run_id+1}/{total}] {run_label} … ",
                                  end="", flush=True)
                            result = _run_one(run_args, n_frames=n_frames,
                                             output_dir=args.output_dir)
                            status = result.get("status", "?")
                            print(f"{status}  ({result.get('wall_time_s', 0):.1f}s)")

                            row = {
                                "run_id": run_id,
                                "scene": scene,
                                "iterations": n_iter,
                                "substeps": n_sub,
                                "dt": dt_val,
                                "ablation": abl_label,
                                "no_contact": no_contact,
                                "no_truncation": no_trunc,
                                "no_friction": no_fric,
                                "diag_path": diag_path,
                                **result,
                            }
                            writer.writerow(row)
                            meta_f.flush()
                            run_id += 1

    print(f"\n[ablation] Done. Metadata → {meta_path}")


if __name__ == "__main__":
    main()
