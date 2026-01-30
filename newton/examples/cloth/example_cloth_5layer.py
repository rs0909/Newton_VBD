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
# Example Cloth Twist
#
# This simulation demonstrates twisting an FEM cloth model using the VBD
# solver, showcasing its ability to handle complex self-contacts while
# ensuring it remains intersection-free.
#
# Command: python -m newton.examples cloth_twist
#
###########################################################################

from newton import data_collector

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


class Example:
    def __init__(self, viewer, args=None):
        # setup simulation parameters first
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        data_collector.record_to_scene("time_step", self.frame_dt)
        
        self.frame_count = 0

        # group related attributes by prefix
        self.sim_time = 0.0
        self.sim_substeps = 10  # must be an even number when using CUDA Graph
        data_collector.record_to_scene("substeps", self.sim_substeps)
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.iterations = 4
        # the BVH used by SolverVBD will be rebuilt every self.bvh_rebuild_frames
        # When the simulated object deforms significantly, simply refitting the BVH can lead to deterioration of the BVH's
        # quality, in this case we need to completely rebuild the tree to achieve better query efficiency.
        self.bvh_rebuild_frames = 10

        self.rot_angular_velocity = math.pi / 3
        self.rot_end_time = 10

        # save a reference to the viewer
        self.viewer = viewer

        usd_stage = Usd.Stage.Open(os.path.join(warp.examples.get_asset_directory(), "square_cloth.usd"))
        usd_prim = usd_stage.GetPrimAtPath("/root/cloth/cloth")

        cloth_mesh = newton.usd.get_mesh(usd_prim)
        mesh_points = cloth_mesh.vertices
        mesh_indices = cloth_mesh.indices

        vertices = [wp.vec3(v) for v in mesh_points]
        self.faces = mesh_indices.reshape(-1, 3)

        scene = newton.ModelBuilder(gravity=-9.8)
        # cloth!
        cloth_colors = wp.array([  
            wp.vec3(0.8, 0.2, 0.2),  # Red  
            wp.vec3(0.2, 0.8, 0.2),  # Green    
            wp.vec3(0.2, 0.2, 0.8),  # Blue  
            wp.vec3(0.8, 0.8, 0.2),  # Yellow  
            wp.vec3(0.8, 0.2, 0.8),  # Magenta  
        ], dtype=wp.vec3)  
        for i in range(0,5):
            scene.add_cloth_mesh( # yk: how to add cloth mesh is written here
                pos=wp.vec3(0.0, 0.0, 1.2+i*0.05),
                rot=wp.quat_from_axis_angle(wp.vec3(1, 0, 0), wp.pi / 2),
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
        # sphere
        print(len(vertices), "<<<<<<<<<<<<< number of points on each cloth")
        
        scene.add_shape_sphere(
            body=-1,
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.6), q=wp.quat_identity()),
            radius=0.5
        )
        # plane
        scene.add_ground_plane()
        scene.color()
        self.model = scene.finalize()
        self.model.soft_contact_ke = 1.0e3
        self.model.soft_contact_kd = 1.0e-4
        self.model.soft_contact_mu = 0.2
        data_collector.record_to_scene("cloth_friction_mu", self.model.soft_contact_mu)

        self.solver = newton.solvers.SolverVBD(
            self.model,
            self.iterations,
            particle_enable_self_contact=True,
            particle_self_contact_radius=0.002,
            particle_self_contact_margin=0.0035, # yk: this handles the entire contact margin. (= query radius)
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Create collision pipeline (default: unified)
        self.collision_pipeline = newton.examples.create_collision_pipeline(self.model, args)
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        self.viewer.set_model(self.model)
        self.viewer.show_contacts = True

        # put graph capture into it's own function
        # self.simulate()
        # self.capture()

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        # yk: how this `collide` function works?
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
        data_collector.record_to_frame("body_cloth_col_count", self.contacts.soft_contact_count.numpy()[0])

        self.solver.rebuild_bvh(self.state_0)
        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model for picking, wind, etc
            self.viewer.apply_forces(self.state_0)

            data_collector.substep_start(substep)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0


    def step(self):
        data_collector.frame_start(self.frame_count)
        data_collector.frame_timer.clear_start()

        # if self.graph:
        #     wp.capture_launch(self.graph)
        # else:
        self.simulate()

        # total_time_end = time.perf_counter()
        # data_collector.record_to_frame("total_time", total_time_end-total_time_start)
        data_collector.frame_timer.stop()
        data_collector.record_to_frame("total_time", data_collector.frame_timer.acc_time)
        self.frame_count += 1

        self.sim_time += self.frame_dt

    def render(self):
        if self.viewer is None:
            return

        # Begin frame with time
        self.viewer.begin_frame(self.sim_time)

        # Render model-driven content (ground plane)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        p_lower = wp.vec3(-0.6, -0.9, -0.6)
        p_upper = wp.vec3(0.6, 0.9, 0.6)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume",
            lambda q, qd: newton.utils.vec_inside_limits(q, p_lower, p_upper),
        )
        newton.examples.test_particle_state(
            self.state_0,
            "particle velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 1.0,
        )


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=300)

    viewer, args = newton.examples.init(parser)

    # Create example and run
    example = Example(viewer, args)

    newton.examples.run(example, args)
