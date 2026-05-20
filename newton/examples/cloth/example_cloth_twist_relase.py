# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Cloth Twist Release (Figure 10, Chen et al. 2025 arXiv:2604.15513)
#
# A square cloth hangs vertically under gravity.  Only the top edge is
# kinematically controlled: it is twisted 6 full rotations over 10 s and
# then released.  The bottom edge is free throughout.  After release the
# cloth untwists naturally under gravity and elastic restoring forces while
# Planar-DAT keeps it intersection-free.
#
# Command: python -m newton.examples cloth_twist_relase
#
###########################################################################

import math
import os

import numpy as np
import warp as wp
import warp.examples
from pxr import Usd

import newton
import newton.examples
import newton.usd
from newton import ParticleFlags


@wp.kernel
def initialize_rotation(
    # input
    vertex_indices_to_rot: wp.array(dtype=wp.int32),
    pos: wp.array(dtype=wp.vec3),
    rot_centers: wp.array(dtype=wp.vec3),
    rot_axes: wp.array(dtype=wp.vec3),
    t: wp.array(dtype=float),
    # output
    roots: wp.array(dtype=wp.vec3),
    roots_to_ps: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    v_index = vertex_indices_to_rot[tid]

    p = pos[v_index]
    rot_center = rot_centers[tid]
    rot_axis = rot_axes[tid]
    op = p - rot_center

    root = wp.dot(op, rot_axis) * rot_axis
    root_to_p = p - root

    roots[tid] = root
    roots_to_ps[tid] = root_to_p

    if tid == 0:
        t[0] = 0.0


@wp.kernel
def apply_top_rotation(
    # input
    vertex_indices_to_rot: wp.array(dtype=wp.int32),
    rot_axes: wp.array(dtype=wp.vec3),
    roots: wp.array(dtype=wp.vec3),
    roots_to_ps: wp.array(dtype=wp.vec3),
    t: wp.array(dtype=float),
    angular_velocity: float,
    dt: float,
    end_time: float,
    # output
    pos_0: wp.array(dtype=wp.vec3),
    pos_1: wp.array(dtype=wp.vec3),
):
    """Kinematically rotate top-edge vertices around Y; stop at end_time."""
    cur_t = t[0]
    if cur_t >= end_time:
        return

    tid = wp.tid()
    v_index = vertex_indices_to_rot[tid]

    rot_axis = rot_axes[tid]
    ux = rot_axis[0]
    uy = rot_axis[1]
    uz = rot_axis[2]

    theta = cur_t * angular_velocity

    R = wp.mat33(
        wp.cos(theta) + ux * ux * (1.0 - wp.cos(theta)),
        ux * uy * (1.0 - wp.cos(theta)) - uz * wp.sin(theta),
        ux * uz * (1.0 - wp.cos(theta)) + uy * wp.sin(theta),
        uy * ux * (1.0 - wp.cos(theta)) + uz * wp.sin(theta),
        wp.cos(theta) + uy * uy * (1.0 - wp.cos(theta)),
        uy * uz * (1.0 - wp.cos(theta)) - ux * wp.sin(theta),
        uz * ux * (1.0 - wp.cos(theta)) - uy * wp.sin(theta),
        uz * uy * (1.0 - wp.cos(theta)) + ux * wp.sin(theta),
        wp.cos(theta) + uz * uz * (1.0 - wp.cos(theta)),
    )

    root = roots[tid]
    root_to_p = roots_to_ps[tid]
    root_to_p_rot = R * root_to_p
    p_rot = root + root_to_p_rot

    pos_0[v_index] = p_rot
    pos_1[v_index] = p_rot

    if tid == 0:
        t[0] = cur_t + dt


class Example:
    def __init__(self, viewer, args=None):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps

        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.iterations = 10
        self.bvh_rebuild_frames = 10

        # 6 full rotations in 10 s (paper Fig. 10)
        self.rot_angular_velocity = 6.0 * 2.0 * math.pi / 10.0
        self.rot_end_time = 10.0
        self.released = False

        self.viewer = viewer

        usd_stage = Usd.Stage.Open(os.path.join(warp.examples.get_asset_directory(), "square_cloth.usd"))
        usd_prim = usd_stage.GetPrimAtPath("/root/cloth/cloth")

        cloth_mesh = newton.usd.get_mesh(usd_prim)
        mesh_points = cloth_mesh.vertices
        mesh_indices = cloth_mesh.indices

        vertices = [wp.vec3(v) for v in mesh_points]
        self.faces = mesh_indices.reshape(-1, 3)

        # Gravity on so the cloth hangs vertically.
        scene = newton.ModelBuilder(gravity=-9.8)
        scene.add_cloth_mesh(
            # rot=90 deg around Z maps the XZ-plane cloth into the YZ plane
            # (original X -> Y), so the cloth spans Y in [-0.25, +0.25].
            # Shift +0.25 m in Y so the top sits near +0.50 m and bottom near 0.
            pos=wp.vec3(0.0, 0.25, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(0, 0, 1), np.pi / 2),
            scale=0.01,
            vertices=vertices,
            indices=mesh_indices,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.2,
            tri_ke=1.0e3,
            tri_ka=1.0e3,
            tri_kd=2.0e-7,
            edge_ke=1e-3,
            edge_kd=1e-4,
        )
        scene.color()
        self.model = scene.finalize()
        self.model.soft_contact_ke = 1.0e3
        self.model.soft_contact_kd = 1.0e-4
        self.model.soft_contact_mu = 0.2

        # After rot=90 deg around Z, the column with the largest original-X
        # becomes the top edge (max Y).  In the 50x50 grid indexed as
        # vertex[i*50+j], column j=49 is the last column (max original-X).
        cloth_size = 50
        top_side = [cloth_size - 1 + i * cloth_size for i in range(cloth_size)]
        # Bottom edge (j=0): only the two corner vertices (i=0 and i=49) are
        # kinematically controlled and rotate in the opposite direction.
        bottom_corners = [0, (cloth_size - 1) * cloth_size]

        # Fix top edge + bottom corners initially.
        flags = self.model.particle_flags.numpy()
        for idx in top_side + bottom_corners:
            flags[idx] = flags[idx] & ~ParticleFlags.ACTIVE
        self.model.particle_flags = wp.array(flags)
        self.top_side = top_side
        self.bottom_corners = bottom_corners

        self.solver = newton.solvers.SolverVBD(
            self.model,
            self.iterations,
            particle_enable_self_contact=True,
            particle_self_contact_radius=0.002,
            particle_self_contact_margin=0.0035,
            ogc_contact=True,
            # Paper (arXiv:2604.15513) Algorithm 2 & 3
            use_planar_dat=True,
            particle_collision_detection_interval=8,
            # Extra mechanisms off for paper-only run
            diagnostic_same_color_pairs=False,
            enable_watchlist=False,
            dynamic_recoloring=False,
            enable_same_color_barrier=False,
            recovery_alpha=0.0,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.collision_pipeline = newton.examples.create_collision_pipeline(self.model, args)
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        # Top edge: +Y rotation; bottom corners: -Y rotation (opposite direction).
        rot_point_indices = top_side + bottom_corners
        rot_axes = [[0, 1, 0]] * len(top_side) + [[0, -1, 0]] * len(bottom_corners)

        self.rot_point_indices = wp.array(rot_point_indices, dtype=int)
        self.t = wp.zeros((1,), dtype=float)
        self.rot_centers = wp.zeros(len(rot_point_indices), dtype=wp.vec3)
        self.rot_axes = wp.array(rot_axes, dtype=wp.vec3)
        self.roots = wp.zeros_like(self.rot_centers)
        self.roots_to_ps = wp.zeros_like(self.rot_centers)

        wp.launch(
            kernel=initialize_rotation,
            dim=self.rot_point_indices.shape[0],
            inputs=[
                self.rot_point_indices,
                self.state_0.particle_q,
                self.rot_centers,
                self.rot_axes,
                self.t,
            ],
            outputs=[self.roots, self.roots_to_ps],
        )

        self.viewer.set_model(self.model)

    def simulate(self):
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
        self.solver.rebuild_bvh(self.state_0)

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)

            # Stop kinematic rotation once released.
            if not self.released:
                wp.launch(
                    kernel=apply_top_rotation,
                    dim=self.rot_point_indices.shape[0],
                    inputs=[
                        self.rot_point_indices,
                        self.rot_axes,
                        self.roots,
                        self.roots_to_ps,
                        self.t,
                        self.rot_angular_velocity,
                        self.sim_dt,
                        self.rot_end_time,
                    ],
                    outputs=[
                        self.state_0.particle_q,
                        self.state_1.particle_q,
                    ],
                )

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        # At rot_end_time release only the bottom corners; top stays fixed.
        if not self.released and self.sim_time >= self.rot_end_time:
            self.released = True
            flags = self.model.particle_flags.numpy()
            for idx in self.bottom_corners:
                flags[idx] = flags[idx] | int(ParticleFlags.ACTIVE)
            self.model.particle_flags = wp.array(flags, device=wp.get_device())

        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        if self.viewer is None:
            return
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        # After 30 s (10 s twist + 20 s untwist) the cloth should still be
        # within a loose bounding volume.
        p_lower = wp.vec3(-0.6, -1.5, -0.6)
        p_upper = wp.vec3(0.6, 0.6, 0.6)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume",
            lambda q, qd: newton.utils.vec_inside_limits(q, p_lower, p_upper),
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=1800)  # 30 s: 10 s twist + 20 s untwist

    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
