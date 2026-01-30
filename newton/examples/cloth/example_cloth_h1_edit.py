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

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

import newton
import newton.examples
import newton.ik as ik
import newton.usd
import newton.utils
from newton import Mesh

from layers.datasets import LayersReader

###############
# SMPL kernel #
###############
import warp as wp

# @wp.kernel
# def smpl_skinning_kernel(
#     # Inputs
#     v_shaped: wp.array(dtype=wp.vec3),      # Vertices after shape blending (T-pose with beta)
#     posedirs: wp.array(dtype=wp.float32, ndim=3), # (6890, 3, 207)
#     weights: wp.array(dtype=wp.float32, ndim=2),  # (6890, 24)
#     G: wp.array(dtype=wp.mat44),            # Global Joint Transforms (24, 4, 4)
#     rot_mats: wp.array(dtype=wp.mat33),     # Rotation matrices for pose blend shapes (24, 3, 3)
    
#     # Outputs
#     v_out: wp.array(dtype=wp.vec3),
# ):
#     tid = wp.tid() # Vertex index

#     # 1. Pose Blend Shapes
#     # Corresponds to: v_posed = v_shaped + posedirs.dot(lrotmin)
#     # lrotmin is (R - I) flattened. 
#     # We iterate through 23 joints (ignore root for pose blend shapes usually)
    
#     v_base = v_shaped[tid]
    
#     # Accumulate pose offsets
#     diff = wp.vec3(0.0, 0.0, 0.0)
    
#     # posedirs shape: [6890, 3, 207]
#     # 207 = 23 joints * 9 elements (3x3 matrix flattened)
#     # SMPL usually skips the root joint (index 0) for pose blend shapes
    
#     for i in range(1, 24): # Iterate joints 1 to 23
#         # Get Rotation matrix (3x3) - Identity
#         # We construct the difference matrix element by element
#         # R = rot_mats[i]
        
#         # Access the rotation matrix columns/rows directly
#         row0 = rot_mats[i][0]
#         row1 = rot_mats[i][1]
#         row2 = rot_mats[i][2]

#         # Subtract Identity (1.0 on diagonal)
#         # 0,0
#         val = row0[0] - 1.0
#         diff += wp.vec3(posedirs[tid, 0, (i-1)*9 + 0], posedirs[tid, 1, (i-1)*9 + 0], posedirs[tid, 2, (i-1)*9 + 0]) * val
#         # 0,1
#         val = row0[1]
#         diff += wp.vec3(posedirs[tid, 0, (i-1)*9 + 1], posedirs[tid, 1, (i-1)*9 + 1], posedirs[tid, 2, (i-1)*9 + 1]) * val
#         # 0,2
#         val = row0[2]
#         diff += wp.vec3(posedirs[tid, 0, (i-1)*9 + 2], posedirs[tid, 1, (i-1)*9 + 2], posedirs[tid, 2, (i-1)*9 + 2]) * val
        
#         # 1,0
#         val = row1[0]
#         diff += wp.vec3(posedirs[tid, 0, (i-1)*9 + 3], posedirs[tid, 1, (i-1)*9 + 3], posedirs[tid, 2, (i-1)*9 + 3]) * val
#         # 1,1
#         val = row1[1] - 1.0
#         diff += wp.vec3(posedirs[tid, 0, (i-1)*9 + 4], posedirs[tid, 1, (i-1)*9 + 4], posedirs[tid, 2, (i-1)*9 + 4]) * val
#         # 1,2
#         val = row1[2]
#         diff += wp.vec3(posedirs[tid, 0, (i-1)*9 + 5], posedirs[tid, 1, (i-1)*9 + 5], posedirs[tid, 2, (i-1)*9 + 5]) * val

#         # 2,0
#         val = row2[0]
#         diff += wp.vec3(posedirs[tid, 0, (i-1)*9 + 6], posedirs[tid, 1, (i-1)*9 + 6], posedirs[tid, 2, (i-1)*9 + 6]) * val
#         # 2,1
#         val = row2[1]
#         diff += wp.vec3(posedirs[tid, 0, (i-1)*9 + 7], posedirs[tid, 1, (i-1)*9 + 7], posedirs[tid, 2, (i-1)*9 + 7]) * val
#         # 2,2
#         val = row2[2] - 1.0
#         diff += wp.vec3(posedirs[tid, 0, (i-1)*9 + 8], posedirs[tid, 1, (i-1)*9 + 8], posedirs[tid, 2, (i-1)*9 + 8]) * val

#     v_posed = v_base + diff

#     # 2. Linear Blend Skinning (LBS)
#     # v_final = Sum(weight_i * G_i * v_posed)
    
#     v_homo = wp.vec4(v_posed[0], v_posed[1], v_posed[2], 1.0)
#     final_pos = wp.vec4(0.0, 0.0, 0.0, 0.0)

#     for j in range(24):
#         w = weights[tid, j]
#         if w > 0.0:
#             # Transform vertex by joint j
#             t_val = G[j] * v_homo
#             final_pos += t_val * w

#     # Divide by w (perspective divide) if necessary, usually w=1 for affine
#     v_out[tid] = wp.vec3(final_pos[0], final_pos[1], final_pos[2])



# # class SMPLSolver:
# #     def __init__(self, layers_reader: LayersReader):
# #         self.num_joint = 24
# #         layers_reader.smpl['female']
# #         self.max_frame = max_frame

# #         self.J_regressor
# #         self.weight
# #         self.posedirs

# #         self.v_template
# #         self.shapedirs
# #         self.faces
# #         self.kintree_table
# #         self.pose_shape = [24, 3]
# #         self.beta_shape = [10]
# #         self.trans_shape = [3]

# #         self.poses = wp.empty(self.max_frame*self.num_joint, dtype=wp.vec3, device=self.device)
# #         self.betas = wp.empty(self.beta_shape)
# #         self.trans = trans


# #     def human_update(self, state_in, state_out):
# #         pass
# #     def garments_update(self, state_in, state_out):
# #         pass
# #     def update(self, state_in, state_out):
# #         self.human_update(state_in, state_out)
# #         self.garments_update(state_in, state_out)

# class SMPLSolver:
#     def __init__(self, layers_reader: LayersReader, sample_name: str, model: newton.Model, device=None):
#         print('Initialize SMPL Solver')
#         self.device = device if device else wp.get_device()
#         self.reader = layers_reader
#         self.sample_name = sample_name
#         self.model = model  # 모델 저장!
        
#         # Load Human Info (Gender/Meta)
#         info = self.reader.read_info(self.sample_name)
#         gender = info['human']['gender']
#         self.smpl_model_np = self.reader.smpl[gender]
#         self.scale = info['human']['scale']

#         # Get Shapes from NumPy model
#         self.num_verts = 6890
#         self.num_joints = 24
        
#         # 1. Pre-load Constant Topology Data to Warp (GPU)
#         # Transpose/Reshape posedirs for kernel: (6890, 3, 207) -> standard
#         print('posedirs shape:', self.smpl_model_np.posedirs.shape)
#         self.d_posedirs = wp.array(self.smpl_model_np.posedirs, dtype=wp.float32, device=self.device)
#         print('weights shape:', self.smpl_model_np.weights.shape)
#         self.d_weights = wp.array(self.smpl_model_np.weights, dtype=wp.float32, device=self.device)
        
#         # 2. Pre-compute Shape Blend Shapes (Optimization)
#         # Since 'beta' is usually constant for a specific character sequence,
#         # we compute v_shaped (T-pose + shape) once on CPU and upload.
#         # This saves doing it every frame on GPU.
#         _, _, shape, _ = self.reader.read_smpl_params(self.sample_name, 0)
#         print('beta shape:', shape)
        
#         # Calculate v_shaped on CPU using existing NumPy logic
#         # v_template + shapedirs * beta
#         v_shaped_np = self.smpl_model_np.v_template + self.smpl_model_np.shapedirs.dot(shape)
#         self.d_v_shaped = wp.array(v_shaped_np, dtype=wp.vec3, device=self.device)
        
#         # 3. Buffers for Dynamic Data
#         self.d_G = wp.array(np.zeros((24, 4, 4)), dtype=wp.mat44, device=self.device)
#         self.d_R = wp.array(np.zeros((24, 3, 3)), dtype=wp.mat33, device=self.device)
#         self.d_v_current = wp.zeros(self.num_verts, dtype=wp.vec3, device=self.device)
#         self.d_v_prev = wp.zeros(self.num_verts, dtype=wp.vec3, device=self.device)
#         self.d_velocity = wp.zeros(self.num_verts, dtype=wp.vec3, device=self.device)

#         # Offsets for Newton State
#         # We assume the Avatar is the first body added, so vertices start at 0
#         self.vertex_offset = 0 
#         self.vertex_count = self.num_verts

#     def update(self, frame_idx, dt): # state_out 인자는 필요 없습니다. Model에 직접 씁니다.
#         """
#         Updates the physics state with the new SMPL pose.
#         """
#         # 1. Read Pose/Trans (그대로)
#         _, pose, _, trans = self.reader.read_smpl_params(self.sample_name, frame_idx)
        
#         # 2. Compute FK (그대로)
#         G, _, root_offset = self.smpl_model_np.set_params(pose=pose, beta=self.smpl_model_np.beta, trans=trans, with_body=False)
#         G[:, :3, 3] *= self.scale
#         R = self.smpl_model_np.R

#         # 3. Upload FK (그대로)
#         self.d_G.assign(G)
#         self.d_R.assign(R)
        
#         # 4. Swap buffers (그대로)
#         wp.copy(self.d_v_prev, self.d_v_current)
        
#         # 5. Launch Skinning Kernel (그대로)
#         wp.launch(
#             kernel=smpl_skinning_kernel,
#             dim=self.num_verts,
#             inputs=[
#                 self.d_v_shaped,
#                 self.d_posedirs,
#                 self.d_weights,
#                 self.d_G,
#                 self.d_R
#             ],
#             outputs=[self.d_v_current],
#             device=self.device
#         )
        
#         # 6. Compute Velocity (그대로)
#         newton.utils.compute_velocity(self.d_v_prev, self.d_v_current, dt, self.d_velocity)

#         # 7. [핵심 수정] Update Newton MODEL Geometry
#         # State(강체 위치)가 아니라 Model(메쉬 모양)을 업데이트해야 합니다.
        
#         # (1) 위치 업데이트: mesh_vertices
#         wp.copy(
#             dest=self.model.mesh_vertices,  # <-- 여기가 핵심 변경 사항
#             src=self.d_v_current, 
#             src_offset=0, 
#             dest_offset=self.vertex_offset, 
#             count=self.vertex_count
#         )
        
#         # (2) 속도 업데이트: mesh_velocities (충돌 반응을 위해 필수)
#         wp.copy(
#             dest=self.model.mesh_velocities, # <-- 여기가 핵심 변경 사항
#             src=self.d_velocity, 
#             src_offset=0, 
#             dest_offset=self.vertex_offset, 
#             count=self.vertex_count
#         )
        
#         # (3) [중요] BVH 리빌드 요청
#         # 메쉬 모양이 바뀌었으므로 충돌 감지 구조를 갱신해야 합니다.
#         self.model.refit_bvh()

# # Add this helper if not present in utils
# @wp.kernel
# def velocity_kernel(
#     pos_prev: wp.array(dtype=wp.vec3),
#     pos_curr: wp.array(dtype=wp.vec3),
#     dt: float,
#     vel_out: wp.array(dtype=wp.vec3)
# ):
#     tid = wp.tid()
#     vel_out[tid] = (pos_curr[tid] - pos_prev[tid]) / dt

# # Monkey patch or add to class
# newton.utils.compute_velocity = lambda prev, curr, dt, out: wp.launch(
#     velocity_kernel, dim=prev.shape[0], inputs=[prev, curr, dt, out], device=prev.device
# )






class Example:
    def __init__(self, viewer, args=None):
        # frame timing
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        # must be an even number when using CUDA Graph
        self.sim_substeps = 6
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.iterations = 20

        self.viewer = viewer
        self.frame_index = 0

        # ------------------------------------------------------------------
        # Build a scene + ground
        # ------------------------------------------------------------------
        asset_path = newton.utils.download_asset("style3d")
        scene = newton.ModelBuilder(gravity=-9.8)

        usd_stage = Usd.Stage.Open(str(asset_path / "avatars" / "Female.usd"))
        usd_prim_avatar = usd_stage.GetPrimAtPath("/Root/Female/Root_SkinnedMesh_Avatar_0_Sub_2")
        avatar_mesh = newton.usd.get_mesh(usd_prim_avatar)
        avatar_mesh_indices = avatar_mesh.indices
        avatar_mesh_points = avatar_mesh.vertices

        layers_reader = LayersReader('/home/user/Desktop/newton/data/demo', '/home/user/Desktop/newton/data/smpl')
        sample_name = '00396'

        pos = wp.vec3(0,0,0.2)
        rot = wp.quat_from_axis_angle(wp.vec3(0,0,1), wp.pi)
        sf = wp.vec3(0.5,0.5,0.5)
        v, f = layers_reader.read_human(sample_name, self.frame_index)
        self.human_mesh = Mesh(v, f)
        scene.add_shape_mesh(
            body=scene.add_body(),
            xform=wp.transform(
                p=pos,
                q=rot,
            ),
            scale=sf,
            mesh=self.human_mesh,
        )
        infos = layers_reader.read_info(sample_name)
        for g in infos['garment']:
            gv, gf, _ = layers_reader.read_garment_vertices_topology(sample_name, g['name'], self.frame_index)
            gf = np.array(gf, dtype=np.int32).flatten()
            scene.add_cloth_mesh(
                pos=pos,
                rot=rot,
                vel=wp.vec3(0.0, 0.0, 0.0),
                tri_ke=1.0e3,
                tri_ka=1.0e3,
                tri_kd=2.0e-7,
                edge_ke=1e-3,
                edge_kd=1e-4,
                density=0.2,
                scale=sf[0],
                vertices=gv,
                indices=gf,
                particle_radius=5e-3
            )

        scene.add_ground_plane()
        scene.color()

        self.graph = None
        self.model = scene.finalize()
        # self.human_mesh.mesh.points *= 2.0

        self.viewer.set_model(self.model) # fix the model?

        # self.smpl_solver = SMPLSolver(layers_reader, sample_name, self.model)

        self.cloth_solver = newton.solvers.SolverVBD(
            self.model,
            self.iterations,
            particle_enable_self_contact=True,
            particle_self_contact_radius=0.002,
            particle_self_contact_margin=0.0035, # yk: this handles the entire contact margin. (= query radius)
            integrate_with_external_rigid_solver=True # yk: this makes the rigid body collision being skipped (but update the contact list)
        )

        self.model.soft_contact_ke = 1.0e3
        self.model.soft_contact_kd = 1.0e-4
        self.model.soft_contact_mu = 0.2

        # states
        self.state0 = self.model.state()
        self.state1 = self.model.state()
        self.control = self.model.control()

        # Create collision pipeline (default: unified)
        self.collision_pipeline = newton.examples.create_collision_pipeline(self.model, args)
        self.contacts = self.model.collide(self.state0, collision_pipeline=self.collision_pipeline)
        self.shape_flags = self.model.shape_flags.numpy()
        
        self.simulate()
        self.capture()

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as cap:
                self.simulate()
            self.graph = cap.graph

    @wp.kernel
    def transform_interpolate(
        ratio: float,
        transform0: wp.array(dtype=wp.transform),
        transform1: wp.array(dtype=wp.transform),
        # outputs
        new_transform: wp.array(dtype=wp.transform),
    ):
        tid = wp.tid()
        tf0 = transform0[tid]
        tf1 = transform1[tid]
        rot0 = wp.transform_get_rotation(tf0)
        rot1 = wp.transform_get_rotation(tf1)
        pos0 = wp.transform_get_translation(tf0)
        pos1 = wp.transform_get_translation(tf1)
        new_pos = wp.lerp(pos0, pos1, ratio)
        new_rot = wp.quat_slerp(rot0, rot1, ratio)
        new_transform[tid] = wp.transformation(new_pos, new_rot, dtype=float)

    def simulate(self):
        # self.smpl_solver.update(self.frame_index, self.frame_dt)

        
        self.contacts = self.model.collide(self.state0, collision_pipeline=self.collision_pipeline)
        self.cloth_solver.rebuild_bvh(self.state0)
        for ii in range(self.sim_substeps):
            self.state0.clear_forces()
            self.cloth_solver.step(self.state0, self.state1, self.control, self.contacts, self.sim_dt)
            (self.state0, self.state1) = (self.state1, self.state0)

        # self.body_q_0.assign(self.body_q_1)

    # def _push_targets_from_gizmos(self):
    #     """Read gizmo-updated transforms and push into IK objectives."""
    #     for i, tf in enumerate(self.ee_tfs):
    #         self.pos_objs[i].set_target_position(0, wp.transform_get_translation(tf))
    #         q = wp.transform_get_rotation(tf)
    #         self.rot_objs[i].set_target_rotation(0, wp.vec4(q[0], q[1], q[2], q[3]))

    # def _force_update_targets(self):
    #     # key infos
    #     key_time = [2.0, 6.0, 10.0]  # second
    #     target_pos = [
    #         [wp.vec3(0.16, 0.65, 1.71), wp.vec3(0.28, -0.50, 1.19)],  # key 0
    #         [wp.vec3(0.12, 0.34, 0.99), wp.vec3(0.14, -0.35, 0.97)],  # key 1
    #     ]
    #     target_rot = [
    #         [wp.quat(0.58, -0.35, 0.29, 0.68), wp.quat(0.00, 0.00, 0.00, 0.00)],  # key 0
    #         [wp.quat(-0.09, 0.46, 0.03, 0.88), wp.quat(-0.09, 0.48, 0.01, 0.87)],  # key 1
    #     ]
    #     if self.sim_time < key_time[0]:
    #         """Raise hands"""
    #         rot_lerp_ratio = wp.clamp(0.3 * self.sim_time / key_time[0], 0.0, 1.0)
    #         pos_lerp_ratio = wp.clamp(0.1 * self.sim_time / key_time[0], 0.0, 1.0)
    #         for i in range(len(target_pos)):
    #             tf = self.ee_tfs[i]
    #             rot = wp.transform_get_rotation(tf)
    #             pos = wp.transform_get_translation(tf)
    #             wp.transform_set_translation(tf, wp.lerp(pos, target_pos[0][i], pos_lerp_ratio))
    #             wp.transform_set_rotation(tf, wp.quat_slerp(rot, target_rot[0][i], rot_lerp_ratio))
    #     elif self.sim_time < key_time[1]:
    #         """Wave hands"""
    #         time_budget = key_time[1] - key_time[0]
    #         rot_angle = wp.sin((self.sim_time - key_time[0]) * 7.5 * wp.pi / time_budget) * 0.3
    #         rot = wp.quat_from_axis_angle(axis=wp.vec3(1, 0, 0), angle=rot_angle) * target_rot[0][0]
    #         pos0 = target_pos[0][0] + wp.vec3(
    #             wp.sin((self.sim_time - key_time[0]) * 7.5 * wp.pi / time_budget) * 0.1, 0.0, 0.0
    #         )
    #         pos1 = target_pos[0][1] + wp.vec3(
    #             0.0, wp.sin((self.sim_time - key_time[0]) * 2.5 * wp.pi / time_budget) * 0.05, 0.0
    #         )
    #         wp.transform_set_rotation(self.ee_tfs[0], wp.quat(rot))
    #         wp.transform_set_translation(self.ee_tfs[0], pos0)
    #         wp.transform_set_translation(self.ee_tfs[1], pos1)
    #     elif self.sim_time < key_time[2]:
    #         """Drop hands"""
    #         pos_lerp_ratio = wp.clamp((self.sim_time - key_time[1]) / (key_time[2] - key_time[1]), 0.0, 1.0)
    #         rot_lerp_ratio = wp.clamp((self.sim_time - key_time[1]) / (key_time[2] - key_time[1]), 0.0, 1.0)
    #         for i in range(len(target_pos)):
    #             tf = self.ee_tfs[i]
    #             rot = wp.transform_get_rotation(tf)
    #             pos = wp.transform_get_translation(tf)
    #             wp.transform_set_translation(tf, wp.lerp(pos, target_pos[1][i], wp.pow(pos_lerp_ratio, 2.0)))
    #             wp.transform_set_rotation(tf, wp.quat_slerp(rot, target_rot[1][i], wp.pow(rot_lerp_ratio, 3.0)))

    # ----------------------------------------------------------------------
    # Template API
    # ----------------------------------------------------------------------
    def step(self):
        if self.frame_index > 0:
            # self._force_update_targets()
            # self._push_targets_from_gizmos()
            if self.graph:
                wp.capture_launch(self.graph)
            elif wp.get_device().is_cuda:
                self.capture()
            else:
                self.simulate()
            self.sim_time += self.frame_dt
        self.frame_index += 1

    def test_final(self):
        p_lower = wp.vec3(-0.3, -0.8, 0.8)
        p_upper = wp.vec3(0.5, 0.8, 1.8)
        newton.examples.test_particle_state(
            self.state0,
            "particles are within a reasonable volume",
            lambda q, qd: newton.utils.vec_inside_limits(q, p_lower, p_upper),
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state0)
        self.viewer.end_frame()
        wp.synchronize()


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=601)
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
