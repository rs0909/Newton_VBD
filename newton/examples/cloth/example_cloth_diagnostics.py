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

###########################################################################
# Cloth Damping Diagnostics Benchmark
#
# Benchmark scenes for quantifying artificial damping in the VBD solver.
# Supports four scenes designed to isolate different damping mechanisms:
#
#   A. no_contact_oscillation   – free oscillation, no self-contact
#   B. frictionless_sliding     – self-contact, mu = 0
#   C. separating_contact       – VT contact that resolves then separates
#   D. twist_release            – twisted cloth released (like arXiv:2604.15513 Fig.10)
#
# Usage:
#   uv run -m newton.examples cloth_diagnostics \
#       --scene no_contact_oscillation \
#       --diagnostics \
#       --diag-output-path output/diag_A.npz \
#       --iterations 100 \
#       --dt 0.001 \
#       --substeps 1
#
###########################################################################

from __future__ import annotations

import argparse
import math
import os

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import ParticleFlags
from newton._src.solvers.vbd.diagnostics import VBDDiagnostics

SCENE_NAMES = ["no_contact_oscillation", "frictionless_sliding", "separating_contact", "twist_release"]


# ---------------------------------------------------------------------------
# Shared grid-cloth factory
# ---------------------------------------------------------------------------

def _build_cloth_grid(
    builder: newton.ModelBuilder,
    dim_x: int = 32,
    dim_y: int = 32,
    cell_size: float = 0.02,
    pos: wp.vec3 = wp.vec3(0.0, 0.0, 0.0),
    rot=None,
    mass: float = 0.05,
    tri_ke: float = 1.0e3,
    tri_ka: float = 1.0e3,
    tri_kd: float = 1.0e-7,
    edge_ke: float = 1.0e-3,
    edge_kd: float = 0.0,
):
    if rot is None:
        rot = wp.quat_identity()
    builder.add_cloth_grid(
        pos=pos,
        rot=rot,
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=dim_x,
        dim_y=dim_y,
        cell_x=cell_size,
        cell_y=cell_size,
        mass=mass,
        tri_ke=tri_ke,
        tri_ka=tri_ka,
        tri_kd=tri_kd,
        edge_ke=edge_ke,
        edge_kd=edge_kd,
    )


# ---------------------------------------------------------------------------
# Scene A: no_contact_oscillation
# ---------------------------------------------------------------------------

def build_scene_no_contact_oscillation(args) -> dict:
    """Free oscillation with two fixed corners, no self-contact.

    Isolates: implicit/local solver under-convergence damping.
    Expected: energy decays proportionally to 1/iterations.
    """
    dim = 24
    builder = newton.ModelBuilder(gravity=0.0 if args.no_gravity else -9.81)
    _build_cloth_grid(
        builder,
        dim_x=dim, dim_y=dim,
        cell_size=0.02,
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi / 2),
        mass=0.05,
        tri_ke=args.stiffness_scale * 1.0e3,
        tri_ka=args.stiffness_scale * 1.0e3,
        tri_kd=1.0e-7,
        edge_ke=args.stiffness_scale * 1.0e-3,
        edge_kd=0.0,
    )
    builder.color(include_bending=True)
    model = builder.finalize()

    # Fix two corners
    flags = model.particle_flags.numpy()
    n = (dim + 1) * (dim + 1)
    for v in [0, dim]:
        flags[v] = flags[v] & ~int(ParticleFlags.ACTIVE)
    model.particle_flags = wp.array(flags)

    # Give cloth an initial out-of-plane displacement to trigger oscillation
    q = model.particle_q.numpy().copy()
    center = q.mean(axis=0)
    dist_from_center = np.linalg.norm(q[:, :2] - center[:2], axis=1)
    max_dist = dist_from_center.max() + 1e-10
    q[:, 2] += 0.1 * np.sin(np.pi * dist_from_center / max_dist)
    model.particle_q = wp.array(q, dtype=wp.vec3)

    model.soft_contact_ke = 1.0e3
    model.soft_contact_kd = 1.0e-4
    model.soft_contact_mu = 0.0

    solver = newton.solvers.SolverVBD(
        model,
        iterations=args.iterations,
        particle_enable_self_contact=False,
        use_cuda_graph=False,
        diagnostics=args._diag,
    )
    return {"model": model, "solver": solver, "has_self_contact": False}


# ---------------------------------------------------------------------------
# Scene B: frictionless_sliding
# ---------------------------------------------------------------------------

def build_scene_frictionless_sliding(args) -> dict:
    """Cloth with self-contact and mu=0 (no friction).

    Isolates: truncation/trust-region damping without friction.
    R_tangent should be large if tangential clipping causes damping.
    """
    dim = 24
    builder = newton.ModelBuilder(gravity=-9.81)
    # Two layers slightly offset to create contact
    _build_cloth_grid(
        builder,
        dim_x=dim, dim_y=dim,
        cell_size=0.02,
        pos=wp.vec3(0.0, 0.02, 0.0),
        rot=wp.quat_identity(),
        mass=0.05,
        tri_ke=args.stiffness_scale * 1.0e3,
        tri_ka=args.stiffness_scale * 1.0e3,
        tri_kd=1.0e-7,
        edge_ke=args.stiffness_scale * 1.0e-3,
        edge_kd=0.0,
    )
    # Second layer directly below
    offset = (dim + 1) * (dim + 1)
    _build_cloth_grid(
        builder,
        dim_x=dim, dim_y=dim,
        cell_size=0.02,
        pos=wp.vec3(0.0, -0.02, 0.0),
        rot=wp.quat_identity(),
        mass=0.05,
        tri_ke=args.stiffness_scale * 1.0e3,
        tri_ka=args.stiffness_scale * 1.0e3,
        tri_kd=1.0e-7,
        edge_ke=args.stiffness_scale * 1.0e-3,
        edge_kd=0.0,
    )
    # Fix corners of second layer
    builder.color(include_bending=True)
    model = builder.finalize()

    flags = model.particle_flags.numpy()
    for v in [offset, offset + dim, offset + (dim+1)*dim, offset + (dim+1)*dim + dim]:
        if v < len(flags):
            flags[v] = flags[v] & ~int(ParticleFlags.ACTIVE)
    model.particle_flags = wp.array(flags)

    radius = 0.004
    margin = 0.005
    mu = 0.0 if args.no_friction else 0.0   # always frictionless for this scene

    model.soft_contact_ke = 1.0e3
    model.soft_contact_kd = 1.0e-4
    model.soft_contact_mu = mu

    solver = newton.solvers.SolverVBD(
        model,
        iterations=args.iterations,
        particle_enable_self_contact=not args.no_contact,
        particle_self_contact_radius=radius,
        particle_self_contact_margin=margin,
        ogc_contact=True,
        use_planar_dat=not args.no_truncation,
        particle_collision_detection_interval=8,
        use_cuda_graph=False,
        diagnostics=args._diag,
    )
    return {"model": model, "solver": solver, "has_self_contact": True}


# ---------------------------------------------------------------------------
# Scene C: separating_contact
# ---------------------------------------------------------------------------

def build_scene_separating_contact(args) -> dict:
    """Single cloth dropped from a small height onto itself (folded).

    Cloth is initialized in a slightly folded (z-tent) shape so that initial
    contact resolves and the cloth separates, exercising the contact pipeline.
    """
    dim = 24
    builder = newton.ModelBuilder(gravity=-9.81)
    _build_cloth_grid(
        builder,
        dim_x=dim, dim_y=dim,
        cell_size=0.02,
        pos=wp.vec3(0.0, 0.0, 0.1),
        rot=wp.quat_identity(),
        mass=0.05,
        tri_ke=args.stiffness_scale * 1.0e3,
        tri_ka=args.stiffness_scale * 1.0e3,
        tri_kd=1.0e-7,
        edge_ke=args.stiffness_scale * 1.0e-3,
        edge_kd=0.0,
    )
    builder.color(include_bending=True)
    model = builder.finalize()

    # Tent shape: vertices in the middle pushed up
    q = model.particle_q.numpy().copy()
    cx, cy = q[:, 0].mean(), q[:, 1].mean()
    r = np.sqrt((q[:, 0] - cx) ** 2 + (q[:, 1] - cy) ** 2)
    max_r = r.max() + 1e-10
    q[:, 2] += 0.04 * (1.0 - r / max_r)
    model.particle_q = wp.array(q, dtype=wp.vec3)

    # Fix one edge
    flags = model.particle_flags.numpy()
    for v in range(dim + 1):
        flags[v] = flags[v] & ~int(ParticleFlags.ACTIVE)
    model.particle_flags = wp.array(flags)

    radius = 0.004
    margin = 0.005
    mu = 0.0 if args.no_friction else 0.3

    model.soft_contact_ke = 1.0e3
    model.soft_contact_kd = 1.0e-4
    model.soft_contact_mu = mu

    solver = newton.solvers.SolverVBD(
        model,
        iterations=args.iterations,
        particle_enable_self_contact=not args.no_contact,
        particle_self_contact_radius=radius,
        particle_self_contact_margin=margin,
        ogc_contact=True,
        use_planar_dat=not args.no_truncation,
        particle_collision_detection_interval=8,
        use_cuda_graph=False,
        diagnostics=args._diag,
    )
    return {"model": model, "solver": solver, "has_self_contact": True}


# ---------------------------------------------------------------------------
# Scene D: twist_release
# ---------------------------------------------------------------------------

@wp.kernel
def _initialize_twist_rotation(
    vertex_indices: wp.array(dtype=wp.int32),
    pos: wp.array(dtype=wp.vec3),
    rot_centers: wp.array(dtype=wp.vec3),
    rot_axes: wp.array(dtype=wp.vec3),
    t: wp.array(dtype=float),
    roots: wp.array(dtype=wp.vec3),
    roots_to_ps: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    v_index = vertex_indices[tid]
    p = pos[v_index]
    rot_center = rot_centers[tid]
    rot_axis = rot_axes[tid]
    op = p - rot_center
    root = wp.dot(op, rot_axis) * rot_axis
    roots[tid] = root
    roots_to_ps[tid] = p - root
    if tid == 0:
        t[0] = 0.0


@wp.kernel
def _apply_twist_rotation(
    vertex_indices: wp.array(dtype=wp.int32),
    rot_axes: wp.array(dtype=wp.vec3),
    roots: wp.array(dtype=wp.vec3),
    roots_to_ps: wp.array(dtype=wp.vec3),
    t: wp.array(dtype=float),
    angular_velocity: float,
    dt: float,
    end_time: float,
    pos_0: wp.array(dtype=wp.vec3),
    pos_1: wp.array(dtype=wp.vec3),
):
    cur_t = t[0]
    if cur_t > end_time * 2.0:
        return
    tid = wp.tid()
    v_index = vertex_indices[tid]
    rot_axis = rot_axes[tid]
    ux, uy, uz = rot_axis[0], rot_axis[1], rot_axis[2]
    if cur_t < end_time:
        theta = cur_t * angular_velocity
    else:
        theta = (end_time - (cur_t - end_time)) * angular_velocity
    cos_t = wp.cos(theta)
    sin_t = wp.sin(theta)
    one_c = 1.0 - cos_t
    R = wp.mat33(
        cos_t + ux*ux*one_c,     ux*uy*one_c - uz*sin_t, ux*uz*one_c + uy*sin_t,
        uy*ux*one_c + uz*sin_t,  cos_t + uy*uy*one_c,    uy*uz*one_c - ux*sin_t,
        uz*ux*one_c - uy*sin_t,  uz*uy*one_c + ux*sin_t, cos_t + uz*uz*one_c,
    )
    p_rot = roots[tid] + R * roots_to_ps[tid]
    pos_0[v_index] = p_rot
    pos_1[v_index] = p_rot
    if tid == 0:
        t[0] = cur_t + dt


def build_scene_twist_release(args) -> dict:
    """Twisted-cloth release (arXiv:2604.15513 Fig. 10 variant).

    Two opposite edges rotate and then are released.  Tests self-contact
    damping under large deformation.
    """
    import warp.examples
    from pxr import Usd
    import newton.usd

    usd_stage = Usd.Stage.Open(
        os.path.join(warp.examples.get_asset_directory(), "square_cloth.usd")
    )
    usd_prim = usd_stage.GetPrimAtPath("/root/cloth/cloth")
    cloth_mesh = newton.usd.get_mesh(usd_prim)
    vertices = [wp.vec3(v) for v in cloth_mesh.vertices]
    mesh_indices = cloth_mesh.indices
    faces = mesh_indices.reshape(-1, 3)

    builder = newton.ModelBuilder(gravity=0)
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_from_axis_angle(wp.vec3(0, 0, 1), np.pi / 2),
        scale=0.01,
        vertices=vertices,
        indices=mesh_indices,
        vel=wp.vec3(0.0, 0.0, 0.0),
        density=0.2,
        tri_ke=args.stiffness_scale * 1.0e3,
        tri_ka=args.stiffness_scale * 1.0e3,
        tri_kd=2.0e-7,
        edge_ke=args.stiffness_scale * 1.0e-3,
        edge_kd=1.0e-4,
    )
    builder.color()
    model = builder.finalize()
    model.soft_contact_ke = 1.0e3
    model.soft_contact_kd = 1.0e-4
    model.soft_contact_mu = 0.0 if args.no_friction else 0.2

    # Find left/right edge vertices (same logic as example_cloth_twist_test.py)
    cloth_size = 50
    left_side = [cloth_size - 1 + i * cloth_size for i in range(cloth_size)]
    right_side = [i * cloth_size for i in range(cloth_size)]
    rot_indices = left_side + right_side

    flags = model.particle_flags.numpy()
    for v in rot_indices:
        if v < len(flags):
            flags[v] = flags[v] & ~int(ParticleFlags.ACTIVE)
    model.particle_flags = wp.array(flags)

    radius = 0.002
    margin = 0.003
    mu = 0.0 if args.no_friction else 0.2

    solver = newton.solvers.SolverVBD(
        model,
        iterations=args.iterations,
        particle_enable_self_contact=not args.no_contact,
        particle_self_contact_radius=radius,
        particle_self_contact_margin=margin,
        ogc_contact=True,
        use_planar_dat=not args.no_truncation,
        particle_collision_detection_interval=8,
        use_cuda_graph=False,
        diagnostics=args._diag,
    )

    rot_axes_np = [[0, 1, 0]] * len(right_side) + [[0, -1, 0]] * len(left_side)
    rot_indices_wp = wp.array(rot_indices, dtype=int)
    t_wp = wp.zeros((1,), dtype=float)
    rot_centers_wp = wp.zeros(len(rot_indices), dtype=wp.vec3)
    rot_axes_wp = wp.array(rot_axes_np, dtype=wp.vec3)
    roots_wp = wp.zeros_like(rot_centers_wp)
    roots_to_ps_wp = wp.zeros_like(rot_centers_wp)

    state_0 = model.state()
    state_1 = model.state()

    wp.launch(
        kernel=_initialize_twist_rotation,
        dim=rot_indices_wp.shape[0],
        inputs=[rot_indices_wp, state_0.particle_q, rot_centers_wp, rot_axes_wp, t_wp],
        outputs=[roots_wp, roots_to_ps_wp],
    )

    return {
        "model": model,
        "solver": solver,
        "has_self_contact": True,
        "state_0": state_0,
        "state_1": state_1,
        "twist_data": {
            "rot_indices": rot_indices_wp,
            "rot_axes": rot_axes_wp,
            "roots": roots_wp,
            "roots_to_ps": roots_to_ps_wp,
            "t": t_wp,
            "angular_velocity": math.pi / 3,
            "end_time": 10.0,
        },
    }


# ---------------------------------------------------------------------------
# Example class
# ---------------------------------------------------------------------------

class Example:
    def __init__(self, viewer, args=None):
        if args is None:
            args = _default_args()

        self.args = args
        self.viewer = viewer

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = args.substeps
        self.sim_dt = args.dt if args.dt > 0 else self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        # Build diagnostics object (None if --diagnostics not set)
        self._diag: VBDDiagnostics | None = None
        if args.diagnostics:
            self._diag = VBDDiagnostics(
                output_path=args.diag_output_path,
                log_iterations=args.diag_log_iterations,
                log_frequency=args.diag_log_frequency,
                log_truncation=True,
            )
            args._diag = self._diag
        else:
            args._diag = None

        # Build scene
        scene_name = args.scene
        if scene_name == "no_contact_oscillation":
            scene = build_scene_no_contact_oscillation(args)
        elif scene_name == "frictionless_sliding":
            scene = build_scene_frictionless_sliding(args)
        elif scene_name == "separating_contact":
            scene = build_scene_separating_contact(args)
        elif scene_name == "twist_release":
            scene = build_scene_twist_release(args)
        else:
            raise ValueError(f"Unknown scene: {scene_name!r}. Choose from {SCENE_NAMES}.")

        self.model = scene["model"]
        self.solver = scene["solver"]
        self._twist_data = scene.get("twist_data")

        if "state_0" in scene:
            self.state_0 = scene["state_0"]
            self.state_1 = scene["state_1"]
        else:
            self.state_0 = self.model.state()
            self.state_1 = self.model.state()

        self.control = self.model.control()
        self.collision_pipeline = newton.examples.create_collision_pipeline(self.model, args)
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        self.viewer.set_model(self.model)

        self._global_substep = 0

    def simulate(self):
        self.contacts = self.model.collide(
            self.state_0, collision_pipeline=self.collision_pipeline
        )

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)

            if self._twist_data is not None:
                td = self._twist_data
                wp.launch(
                    kernel=_apply_twist_rotation,
                    dim=td["rot_indices"].shape[0],
                    inputs=[
                        td["rot_indices"], td["rot_axes"], td["roots"],
                        td["roots_to_ps"], td["t"],
                        td["angular_velocity"], self.sim_dt, td["end_time"],
                    ],
                    outputs=[self.state_0.particle_q, self.state_1.particle_q],
                )

            if self._diag is not None:
                self._diag.begin_step(
                    step_num=self._global_substep // self.sim_substeps,
                    substep_num=self._global_substep,
                    dt=self.sim_dt,
                )

            self.solver.step(
                self.state_0, self.state_1, self.control, self.contacts, self.sim_dt
            )
            self._global_substep += 1

            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        pass

    def __del__(self):
        if self._diag is not None:
            self._diag.save()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_args():
    """Fallback args with all defaults when running without argparse."""
    import types
    a = types.SimpleNamespace()
    a.scene = "no_contact_oscillation"
    a.diagnostics = False
    a.diag_output_path = "output/diagnostics.npz"
    a.diag_log_iterations = False
    a.diag_log_frequency = 1
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


def add_diagnostics_args(parser: argparse.ArgumentParser) -> None:
    grp = parser.add_argument_group("diagnostics")
    grp.add_argument("--scene", choices=SCENE_NAMES, default="no_contact_oscillation",
                     help="Benchmark scene to run.")
    grp.add_argument("--diagnostics", action="store_true",
                     help="Enable VBDDiagnostics and save energy/momentum logs.")
    grp.add_argument("--diag-output-path", default="output/diagnostics.npz",
                     help="Path for .npz diagnostics output.")
    grp.add_argument("--diag-log-iterations", action="store_true",
                     help="Log kinetic energy / residual after every VBD iteration (slow; disables CUDA graph).")
    grp.add_argument("--diag-log-frequency", type=int, default=1,
                     help="Record every N substeps (default: 1 = every substep).")

    grp2 = parser.add_argument_group("solver")
    grp2.add_argument("--iterations", type=int, default=100,
                      help="VBD iterations per substep.")
    grp2.add_argument("--dt", type=float, default=-1.0,
                      help="Simulation timestep in seconds. -1 = frame_dt / substeps.")
    grp2.add_argument("--substeps", type=int, default=10,
                      help="Substeps per rendered frame.")
    grp2.add_argument("--no-contact", action="store_true",
                      help="Disable particle self-contact (ablation).")
    grp2.add_argument("--no-truncation", action="store_true",
                      help="Disable Planar-DAT truncation (ablation).")
    grp2.add_argument("--no-friction", action="store_true",
                      help="Set friction mu=0 (ablation).")
    grp2.add_argument("--no-gravity", action="store_true",
                      help="Disable gravity (for scene A oscillation).")
    grp2.add_argument("--stiffness-scale", type=float, default=1.0,
                      help="Multiply all stiffness values by this factor.")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    add_diagnostics_args(parser)
    viewer, args = newton.examples.init(parser)

    example = Example(viewer=viewer, args=args)
    newton.examples.run(example, args)

    if example._diag is not None:
        example._diag.save()
