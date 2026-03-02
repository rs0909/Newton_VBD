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

@wp.kernel
def transform_points(
    # inputs
    in_points: wp.array(dtype=wp.vec3),
    xform: wp.transform,
    # outputs
    out_points: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    out_points[tid] = wp.transform_point(xform, in_points[tid])

@wp.kernel
def transform_vectors(
    # inputs
    in_vectors: wp.array(dtype=wp.vec3),
    xform: wp.transform,
    # outputs
    out_vectors: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    out_vectors[tid] = wp.transform_vector(xform, in_vectors[tid])

class SMPLMesh(DynamicMesh):
    def __init__(self, layers_reader: LayersReader, sample_name, viewer_name, sub_steps, device, xform=None, scale=None):
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

        self.analytic_velocities = wp.empty_like(self.curr_pos)

        # etc
        self.xform = xform
        self.scale = scale

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
            self.analytic_velocities = (self.next_pos-self.curr_pos)/dt
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
        self.update_vertices(self.new_position, dt, self.analytic_velocities)
        viewer.update_mesh(self.viewer_name, self.new_position, None, None, None)
        
        self.acc_sub_steps += 1

    def world_positions(self, positions):
        p = wp.empty_like(positions)
        wp.launch(
            kernel=transform_points,
            dim=self.num_vertices,
            inputs=[
                positions*self.scale,
                self.xform,
            ],
            outputs=[
                p
            ]
        )
        return p

    def world_velocities(self, velocities):
        v = wp.empty_like(velocities)
        wp.launch(
            kernel=transform_vectors,
            dim=self.num_vertices,
            inputs=[
                velocities,
                self.xform,
            ],
            outputs=[
                v
            ]
        )
        return v

    def debug_velocities(self, viewer, vis_vel_scale = None):
        if self.xform is None or self.scale is None:
            print('No xform or scale is given to debug velocities')
            return
        if vis_vel_scale is None:
            vis_vel_scale = 1.0 / self.sub_steps

        points = self.world_positions(self.points_buffer)
        velocities = self.world_velocities(self.velocities_buffer)
        viewer.log_lines('/debug/mesh_velocities', points, points+velocities*vis_vel_scale, (1, 0, 0))

def debug_contact_points(name, viewer, contact_points, color=(0,1,0), radii=0.05, device=wp.get_device()):
    points = wp.array(contact_points, dtype=wp.vec3, device=device)
    radius = wp.empty(len(points), dtype=float, device=device)
    radius.fill_(radii)
    colors = wp.empty(len(points), dtype=wp.vec3, device=device)
    colors.fill_(color)
    viewer.log_points(name, points, radius, colors)



@wp.kernel
def check_particle_valid(
    # input
    pos: wp.array(dtype=wp.vec3),
    p_inv_mass: wp.array(dtype=wp.float32),
    conservative_bound: wp.array(dtype=float),
    conservative_bound_eps: float,
    # output
    fixed_counter: wp.array(dtype=int),
    fixed_particle: wp.array(dtype=wp.vec3)
):
    tid = wp.tid()
    p = pos[tid]
    if wp.isnan(p[0]) or wp.isnan(p[1]) or wp.isnan(p[2]):
        wp.printf("ERROR: Particle %d position is NaN!\n", tid)
    if p_inv_mass[tid] < 1e-6:
        wp.printf("INFO: Particle %d is pinned (inv_mass ~ 0)\n", tid)
    if conservative_bound[tid] < conservative_bound_eps:
        # wp.printf("INFO: Particle %d is pinned (conservative_bound ~ 0)\n", tid)
        idx = wp.atomic_add(fixed_counter, 0, 1)
        fixed_particle[idx] = p


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
        self.iterations = 20

        self.viewer = viewer
        self.frame_index = 0

        # ------------------------------------------------------------------
        # Build a scene + ground
        # ------------------------------------------------------------------
        self.layers_reader = LayersReader('/home/user/Desktop/newton/data/demo', '/home/user/Desktop/newton/data/smpl')
        # self.sample_name = '00396'
        # self.sample_name = '00023'
        self.sample_name = '00041'
        # self.sample_name = '00084'
        # self.sample_name = '00119'

        viewer_name = f"/geometry/mesh_0"
        slowdown_factor = self.fps/self.sim_substeps
        self.human_mesh = SMPLMesh(
            layers_reader=self.layers_reader, 
            sample_name=self.sample_name, 
            viewer_name=viewer_name, 
            sub_steps=self.sim_substeps * slowdown_factor, 
            # sub_steps=self.sim_substeps,
            device=wp.get_device(),
        )



        # scale
        # self.sf = 50.0
        self.sf = 10.0
        sf = self.sf
        uniform_sf = wp.vec3(sf,sf,sf)

        # position
        v = self.human_mesh.curr_pos.numpy()
        human_mean_v = wp.vec3(v.mean(axis=0)) * sf
        human_minz_v = v[:, 2].min() * sf
        human_maxz_v = v[:, 2].max() * sf
        human_z_size = human_maxz_v-human_minz_v
        d = human_z_size * 2
        pos = wp.vec3(0.0,0.0,-human_minz_v)

        # rotations
        rot_dict = {
            '00396': wp.quat_from_axis_angle(wp.vec3(0,0,1), wp.pi),
            '00023': wp.quat_identity(),
            '00041': wp.quat_from_axis_angle(wp.vec3(0,0,1), wp.pi/2.0),
            '00119': wp.quat_from_axis_angle(wp.vec3(0,0,1), -wp.pi/2.0),
        }
        if self.sample_name not in rot_dict:
            rot = wp.quat_identity()
        else:
            rot = rot_dict[self.sample_name]

        # camera position
        cam_pos = wp.transform_point(wp.transform(p=wp.vec3(0,0,0),q=rot), human_mean_v) + wp.vec3(d, 0, 0)
        
        # scene setting
        scene = newton.ModelBuilder(gravity=-9.8*sf)
        human_hidden = False
        if human_hidden:
            self.human_mesh = None
        else:
            self.human_mesh_xform = wp.transform(p=pos, q=rot)
            self.human_mesh.xform = self.human_mesh_xform
            self.human_mesh.scale = sf
            scene.add_shape_mesh(
                body=scene.add_body(),
                xform=self.human_mesh_xform,
                scale=uniform_sf,
                mesh=self.human_mesh,
            )

        # control the visibility cloth layers
        g_visible_list = [0] # start from 0

        infos = self.layers_reader.read_info(self.sample_name)
        for i, g in enumerate(infos['garment']):
            if i not in g_visible_list:
                continue

            gv, gf, _ = self.layers_reader.read_garment_vertices_topology(self.sample_name, g['name'], self.frame_index)
            gf = np.array(gf, dtype=np.int32).flatten()
            scene.add_cloth_mesh(
                pos=pos,
                rot=rot,
                vel=wp.vec3(0.0, 0.0, 0.0),
                tri_ke=1.0e5,
                tri_ka=1.0e5,
                tri_kd=2.0e-7, # initial
                # tri_kd=2.0e0,
                edge_ke=1e-3,
                edge_kd=1e-4,
                density=0.2,
                scale=uniform_sf[0],
                vertices=gv,
                indices=gf,
                # particle_radius=5e-3
                particle_radius=0.10 * sf*2 # for rigid-particle collision
                # particle_radius=5e-2 * sf*2
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
            particle_self_contact_radius=0.1*sf/2.0, # particle
            particle_self_contact_margin=0.2*sf/2.0, # yk: this handles the entire contact margin. (= query radius)
            integrate_with_external_rigid_solver=True, # yk: this makes the rigid body collision being skipped (but update the contact list)
            rigid_contact_k_start=1.0e5,
            ogc_contact=True
        )

        self.model.soft_contact_ke = 1.0e5
        self.model.soft_contact_kd = 1.0e-4 # initial
        # self.model.soft_contact_kd = 1.0e1 # modified
        self.model.soft_contact_mu = 0.5
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
        #     self.human_mesh.update(self.viewer, self.sim_dt)

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
                # human update
                self.human_mesh.update(self.viewer, self.sim_dt)

                # self.human_mesh.debug_velocities(self.viewer)

                # collision detection again
                self.contacts = self.model.collide(self.state0, collision_pipeline=self.collision_pipeline)

                soft_col_count = self.contacts.soft_contact_count.numpy()[0]
                # debug_contact_points('/debug/soft_contact_points', self.viewer, self.contacts.soft_contact_body_pos.numpy()[:soft_col_count], (0, 1, 0), 0.05*self.sf)

            self.cloth_solver.step(self.state0, self.state1, self.control, self.contacts, self.sim_dt)
            


            fixed_counter = wp.zeros(1, dtype=int, device=wp.get_device())
            fixed_particle_positions = wp.empty(self.state0.particle_count, dtype=wp.vec3, device=wp.get_device())
                    
            wp.launch(
                kernel=check_particle_valid,
                dim=self.state0.particle_count,
                inputs=[
                    self.state0.particle_q,
                    self.model.particle_inv_mass,
                    self.cloth_solver.particle_conservative_bounds,
                    1e-3,
                ],
                outputs=[
                    fixed_counter,
                    fixed_particle_positions
                ]
            )

            fixed_num = fixed_counter.numpy()[0]
            debug_contact_points('/debug/fixed_points', self.viewer, fixed_particle_positions.numpy()[:fixed_num], (1, 0, 0), 0.05*self.sf)

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
