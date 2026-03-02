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
# Example cloth H1 (cloth-robot interaction)
#
# The H1 robot in a jacket waves hello to us, powered by the Style3D solver
# for cloth and driven by an IkSolver for robot kinematics.
#
# Demonstrates how to leverage interpolated robot kinematics within the
# collision processing pipeline and feed the results to the cloth solver.
#
# Command: python -m newton.examples cloth_h1
#
###########################################################################

from newton import data_collector

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

import newton
import newton.examples
import newton.ik as ik
import newton.usd
import newton.utils
from newton import Mesh, DynamicMesh

from layers.datasets import LayersReader

import warp as wp

@wp.kernel
def linear_interpolation(
    # inputs
    curr_pos: wp.array(dtype=wp.vec3),
    next_pos: wp.array(dtype=wp.vec3),
    ratio: float,
    # outputs
    new_position: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    new_position[tid] = curr_pos[tid] * (1.0-ratio) + next_pos[tid] * ratio


class SMPLMesh(DynamicMesh):
    def __init__(self, layers_reader: LayersReader, sample_name, viewer_name, sub_steps, device):
        self.device = device
        # reader info
        self.layers_reader = layers_reader
        self.sample_name = sample_name

        # animation info
        self.viewer_name = viewer_name
        self.sub_steps = sub_steps
        self.max_frame = self.layers_reader.read_info(self.sample_name)['human']['seq_end']
        self.total_sub_steps = self.max_frame * self.sub_steps
        self.curr_frame = 0
        self.next_frame = 1
        v, f = self.layers_reader.read_human(self.sample_name, self.curr_frame)
        super().__init__(v, f)
        self.curr_pos = wp.array(v, dtype=wp.vec3, device=self.device)
        v, _ = self.layers_reader.read_human(self.sample_name, self.next_frame)
        self.next_pos = wp.array(v, dtype=wp.vec3, device=self.device)
        self.new_position = wp.empty_like(self.curr_pos, device=self.device)
        self.num_vertices = len(self.curr_pos)
        self.acc_sub_steps = 0

    def update(self, viewer, dt):
        if self.acc_sub_steps >= self.total_sub_steps:
            return
        
        query_frame = self.acc_sub_steps // self.sub_steps
        if query_frame != self.curr_frame:
            self.curr_frame = query_frame
            self.next_frame = query_frame+1
            v, _ = self.layers_reader.read_human(self.sample_name, self.curr_frame)
            self.curr_pos = wp.array(v, dtype=wp.vec3, device=self.device)
            v, _ = self.layers_reader.read_human(self.sample_name, self.next_frame)
            self.next_pos = wp.array(v, dtype=wp.vec3, device=self.device)
        query_ratio = float(self.acc_sub_steps % self.sub_steps) / float(self.sub_steps)

        wp.launch(
            linear_interpolation,
            dim=self.num_vertices,
            inputs=[
                self.curr_pos,
                self.next_pos,
                query_ratio,
            ],
            outputs=[
                self.new_position,
            ]
        )
        self.update_vertices(self.new_position, dt)
        viewer.update_mesh(self.viewer_name, self.new_position, None, None, None)
        
        self.acc_sub_steps += 1





class Example:
    def __init__(self, viewer, args=None):
        data_collector.set_record_name('SMPL_2layers_00084_test')
        # frame timing
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        data_collector.record_to_scene("time_step", self.frame_dt)
        self.sim_time = 0.0

        # must be an even number when using CUDA Graph
        self.sim_substeps = 6
        data_collector.record_to_scene("substeps", self.sim_substeps)
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.iterations = 30

        self.viewer = viewer
        self.frame_index = 0

        # ------------------------------------------------------------------
        # Build a scene + ground
        # ------------------------------------------------------------------
        # asset_path = newton.utils.download_asset("style3d")
        scene = newton.ModelBuilder(gravity=-9.8)

        # usd_stage = Usd.Stage.Open(str(asset_path / "avatars" / "Female.usd"))
        # usd_prim_avatar = usd_stage.GetPrimAtPath("/Root/Female/Root_SkinnedMesh_Avatar_0_Sub_2")
        # avatar_mesh = newton.usd.get_mesh(usd_prim_avatar)
        # avatar_mesh_indices = avatar_mesh.indices
        # avatar_mesh_points = avatar_mesh.vertices

        self.layers_reader = LayersReader('/home/user/Desktop/newton/data/demo', '/home/user/Desktop/newton/data/smpl')
        # self.sample_name = '00396'
        # self.sample_name = '00023'
        self.sample_name = '00041'
        # self.sample_name = '00084'
        # self.sample_name = '00119'
        # self.max_frame = self.layers_reader.read_info(self.sample_name)['human']['seq_end']

        v, f = self.layers_reader.read_human(self.sample_name, self.frame_index)
        # self.human_mesh = DynamicMesh(v, f)

        viewer_name = f"/geometry/mesh_0"
        slowdown_factor = 10
        # self.human_mesh = SMPLMesh(
        #     layers_reader=self.layers_reader, 
        #     sample_name=self.sample_name, 
        #     viewer_name=viewer_name, 
        #     sub_steps=self.sim_substeps * slowdown_factor, 
        #     device=wp.get_device(),
        # )
        self.human_mesh = DynamicMesh(v, f)



        # scale
        sf = 10.0
        uniform_sf = wp.vec3(sf,sf,sf)

        # position
        # v = self.human_mesh.curr_pos.numpy()
        human_mean_v = wp.vec3(v.mean(axis=0)) * sf
        human_minz_v = v[:, 2].min() * sf
        human_maxz_v = v[:, 2].max() * sf
        d = (human_maxz_v-human_minz_v) * 2
        pos = wp.vec3(0.0,0.0,-human_minz_v)

        # rotations
        rot_dict = {
            # '00396': wp.quat_from_axis_angle(wp.vec3(0,0,1), wp.pi),
            # '00023': wp.quat_identity(),
            # '00041': wp.quat_from_axis_angle(wp.vec3(0,0,1), wp.pi/2.0),
            # '00119': wp.quat_from_axis_angle(wp.vec3(0,0,1), -wp.pi/2.0),
        }
        if self.sample_name not in rot_dict:
            rot = wp.quat_identity()
        else:
            rot = rot_dict[self.sample_name]

        # camera position
        cam_pos = wp.transform_point(wp.transform(p=wp.vec3(0,0,0),q=rot), human_mean_v) + wp.vec3(d, 0, 0)

        # self.human_mesh = None
        scene.add_shape_mesh(
            body=scene.add_body(),
            xform=wp.transform(
                p=pos,
                q=rot,
            ),
            scale=uniform_sf,
            mesh=self.human_mesh,
        )

        # control the visibility cloth layers
        
        vertices = []
        num_particles = 50
        size = 20.0 * sf
        offset = size/float(num_particles)
        center = wp.vec3(human_mean_v[0], human_mean_v[1], human_maxz_v) + wp.vec3(0, 0, size)
        for z in range(num_particles):
            for y in range(num_particles):
                vertices.append(wp.vec3(0, y*offset-size/2.0, z*offset-size/2.0))

        vertices2 = []
        center2 = wp.vec3(human_mean_v[0], human_mean_v[1], human_maxz_v) + wp.vec3(0, 0, 1)*sf
        for x in range(num_particles):
            for y in range(num_particles):
                vertices2.append(wp.vec3(x*offset-size/2.0, y*offset-size/2.0, 0))
        
        mesh_indices = []
        for z in range(num_particles-1):
            for y in range(num_particles-1):
                pivot = z*num_particles + y
                mesh_indices.append(pivot)
                mesh_indices.append(pivot + num_particles)
                mesh_indices.append(pivot + 1)
                mesh_indices.append(pivot + 1)
                mesh_indices.append(pivot + num_particles)
                mesh_indices.append(pivot + num_particles + 1)

        scene.add_cloth_mesh( # yk: how to add cloth mesh is written here
            pos=wp.vec3(0.0, 0.0, 0.0)+center2,
            # rot=wp.quat_from_axis_angle(wp.vec3(0, 0, 1), wp.pi / 2), # YZ plane
            rot=wp.quat_identity(), # XZ plane
            # rot=wp.quat_from_axis_angle(wp.vec3(0,0,1), wp.pi/4), # X=Y plane
            # rot=wp.quat_from_axis_angle(wp.vec3(1,0,0), wp.pi/4),
            scale=1,
            vertices=vertices2,
            indices=mesh_indices,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.2,
            tri_ke=1.0e3,
            tri_ka=1.0e3,
            tri_kd=2.0e-7,
            edge_ke=1e-3,
            edge_kd=1e-4,
            # particle_radius=0.1*sf*2
            particle_radius=0.1*sf*2
        )

        scene.add_ground_plane()
        scene.color()

        self.graph = None
        self.model = scene.finalize()

        self.viewer.set_model(self.model) # fix the model?
        self.viewer.set_camera(
            pos=cam_pos,
            pitch=0.0,
            yaw=-180.0,
        )
        self.viewer.camera.near = max(1.0, d*0.1)
        self.viewer.camera.far = max(1000.0, d*1.5)
        self.viewer._cam_speed = 4.0 * sf
        print()
        print(f">>>>>>>example init>>>>>>>>>>>> {self.viewer.objects}")
        print()

        self.cloth_solver = newton.solvers.SolverVBD(
            self.model,
            self.iterations,
            particle_enable_self_contact=True,
            particle_self_contact_radius=0.002 *sf*2, # self collision
            particle_self_contact_margin=0.0035*sf*2, # yk: this handles the entire contact margin. (= query radius)
            integrate_with_external_rigid_solver=True # yk: this makes the rigid body collision being skipped (but update the contact list)
        )

        self.model.soft_contact_ke = 1.0e3
        self.model.soft_contact_kd = 1.0e-4
        self.model.soft_contact_mu = 0.2
        data_collector.record_to_scene("cloth_friction_mu", self.model.soft_contact_mu)

        # states
        self.state0 = self.model.state()
        self.state1 = self.model.state()
        self.control = self.model.control()

        # Create collision pipeline (default: unified)
        self.collision_pipeline = newton.examples.create_collision_pipeline(self.model, args)
        self.contacts = self.model.collide(self.state0, collision_pipeline=self.collision_pipeline)
        self.shape_flags = self.model.shape_flags.numpy()

        self.log_once = True
        
        # self.simulate()
        # self.capture()

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as cap:
                self.simulate()
            self.graph = cap.graph

    def simulate(self):
        # if self.human_mesh:
        #     self.human_mesh.update_vertices(self.human_mesh.points_buffer + wp.vec3(0,0,0.00005), self.sim_dt)

        # simulate!
        self.contacts = self.model.collide(self.state0, collision_pipeline=self.collision_pipeline)
        data_collector.record_to_frame("body_cloth_col_count", self.contacts.soft_contact_count.numpy()[0])


        # print()
        # print(f">>>>>>>>>simulation>>>>>>> {self.viewer.objects}")
        # print()

        self.cloth_solver.rebuild_bvh(self.state0)
        for ii in range(self.sim_substeps):
            self.state0.clear_forces()
            data_collector.substep_start(ii)
            if self.human_mesh:
                self.human_mesh.update_vertices(self.human_mesh.points_buffer + wp.vec3(0,0,0.00005), self.sim_dt)
                self.contacts = self.model.collide(self.state0, collision_pipeline=self.collision_pipeline)
                # debugging soft contacts number
                # softcolcount = self.contacts.soft_contact_count.numpy()[0]
                # if softcolcount != 0:
                #     print(f">>>>>>>> frame {self.frame_index}, {ii}: {softcolcount}")
            self.cloth_solver.step(self.state0, self.state1, self.control, self.contacts, self.sim_dt)
            (self.state0, self.state1) = (self.state1, self.state0)

    # ----------------------------------------------------------------------
    # Template API
    # ----------------------------------------------------------------------
    def step(self):
        data_collector.frame_start(self.frame_index)
        data_collector.frame_timer.clear_start()

        # if self.frame_index > 0:
            # if self.graph:
            #     wp.capture_launch(self.graph)
            # elif wp.get_device().is_cuda:
            #     self.capture()
            # else:
        self.simulate()
        self.sim_time += self.frame_dt
        data_collector.frame_timer.stop()
        data_collector.record_to_frame("total_time", data_collector.frame_timer.acc_time)
        
        self.frame_index += 1


    def render(self):
        self.viewer.begin_frame(self.sim_time)

        if self.log_once:
            print()
            print(f">>>>>>>>>before log state>>>>>>> {self.viewer.objects}")
            print()
        self.viewer.log_state(self.state0)

        if self.log_once:
            print()
            print(f">>>>>>>>>after log state>>>>>>> {self.viewer.objects}")
            print()
            self.log_once = False
        self.viewer.end_frame()
        wp.synchronize()


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=601)
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
