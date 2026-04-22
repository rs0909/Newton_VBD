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

"""Particle/soft-body VBD helper routines.

This module is intended to host the particle/soft-body specific parts of the
VBD solver (cloth, springs, triangles, tets, particle contacts, etc.).

The high-level :class:`SolverVBD` interface should remain in
``solver_vbd.py`` and call into functions defined here.
"""

from __future__ import annotations

import numpy as np
import warp as wp
from warp.types import float32, matrix

from newton._src.solvers.vbd.rigid_vbd_kernels import evaluate_body_particle_contact, evaluate_body_particle_contact_log_collision

from ...geometry import ParticleFlags
from ...geometry.kernels import triangle_closest_point
from .tri_mesh_collision import TriMeshCollisionInfo

# TODO: Grab changes from Warp that has fixed the backward pass
wp.set_module_options({"enable_backward": False})

VBD_DEBUG_PRINTING_OPTIONS = {
    # "elasticity_force_hessian",
    # "contact_force_hessian",
    # "contact_force_hessian_vt",
    # "contact_force_hessian_ee",
    # "overall_force_hessian",
    # "inertia_force_hessian",
    # "connectivity",
    # "contact_info",
}

NUM_THREADS_PER_COLLISION_PRIMITIVE = 4
TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE = 16


class mat32(matrix(shape=(3, 2), dtype=float32)):
    pass

def compute_stvk_hessian_blocks_rest(tri_id, pos_rest, tri_indices, tri_poses, tri_areas, tri_materials):
    """
    삼각형 하나에서 모든 vertex 쌍의 Hessian 블록 계산
    반환: H[i][j] = K_ij (3×3) for i,j in {0,1,2}
    """
    v0, v1, v2 = tri_indices[tri_id]
    x0 = pos_rest[v0]
    x01 = pos_rest[v1] - x0
    x02 = pos_rest[v2] - x0

    DmInv = tri_poses[tri_id]  # 2×2
    DmInv00, DmInv01 = DmInv[0, 0], DmInv[0, 1]
    DmInv10, DmInv11 = DmInv[1, 0], DmInv[1, 1]

    mu = tri_materials[tri_id, 0]
    lmbd = tri_materials[tri_id, 1]
    area = tri_areas[tri_id]

    # Deformation gradient (rest-shape에서는 F = I에 가까움)
    f0 = x01 * DmInv00 + x02 * DmInv10
    f1 = x01 * DmInv01 + x02 * DmInv11

    f0_dot_f0 = np.dot(f0, f0)
    f1_dot_f1 = np.dot(f1, f1)
    f0_dot_f1 = np.dot(f0, f1)

    I3 = np.eye(3)
    f0_outer_f0 = np.outer(f0, f0)
    f1_outer_f1 = np.outer(f1, f1)
    f0_outer_f1 = np.outer(f0, f1)
    f1_outer_f0 = np.outer(f1, f0)

    Ic = f0_dot_f0 + f1_dot_f1
    two_dpsi_dIc = -mu + (0.5 * Ic - 1.0) * lmbd

    # 삼각형 공통 항 (d2E/dF2)
    A00 = lmbd * f0_outer_f0 + two_dpsi_dIc * I3 + mu * (f0_dot_f0 * I3 + 2.0 * f0_outer_f0 + f1_outer_f1)
    A11 = lmbd * f1_outer_f1 + two_dpsi_dIc * I3 + mu * (f1_dot_f1 * I3 + 2.0 * f1_outer_f1 + f0_outer_f0)
    A01 = lmbd * f0_outer_f1 + mu * (f0_dot_f1 * I3 + f1_outer_f0)
    A01_sym = A01 + A01.T

    # 각 vertex의 df0_dx, df1_dx 스칼라
    # v_order: 0=v0, 1=v1, 2=v2
    df0 = np.array([
        DmInv00 * (-1) + DmInv10 * (-1),  # v0: mask1-mask0=-1, mask2-mask0=-1
        DmInv00 * ( 1) + DmInv10 * ( 0),  # v1: mask1-mask0=+1, mask2-mask0= 0
        DmInv00 * ( 0) + DmInv10 * ( 1),  # v2: mask1-mask0= 0, mask2-mask0=+1
    ])
    df1 = np.array([
        DmInv01 * (-1) + DmInv11 * (-1),
        DmInv01 * ( 1) + DmInv11 * ( 0),
        DmInv01 * ( 0) + DmInv11 * ( 1),
    ])

    # 모든 vertex 쌍의 K_ij 계산
    H = np.zeros((3, 3, 3, 3))  # H[i][j] = 3×3 행렬
    for i in range(3):
        for j in range(3):
            H[i, j] = (df0[i]*df0[j] * A00
                     + df1[i]*df1[j] * A11
                     + df0[i]*df1[j] * A01
                     + df1[i]*df0[j] * A01.T) * area
    return H, [v0, v1, v2]

@wp.struct
class ParticleForceElementAdjacencyInfo:
    r"""
    - vertex_adjacent_[element]: the flatten adjacency information. Its size is \sum_{i\inV} 2*N_i, where N_i is the
    number of vertex i's adjacent [element]. For each adjacent element it stores 2 information:
        - the id of the adjacent element
        - the order of the vertex in the element, which is essential to compute the force and hessian for the vertex
    - vertex_adjacent_[element]_offsets: stores where each vertex information starts in the  flatten adjacency array.
    Its size is |V|+1 such that the number of vertex i's adjacent [element] can be computed as
    vertex_adjacent_[element]_offsets[i+1]-vertex_adjacent_[element]_offsets[i].
    """

    v_adj_faces: wp.array(dtype=int)
    v_adj_faces_offsets: wp.array(dtype=int)

    v_adj_edges: wp.array(dtype=int)
    v_adj_edges_offsets: wp.array(dtype=int)

    v_adj_springs: wp.array(dtype=int)
    v_adj_springs_offsets: wp.array(dtype=int)

    def to(self, device):
        if device == self.v_adj_faces.device:
            return self
        else:
            adjacency_gpu = ParticleForceElementAdjacencyInfo()
            adjacency_gpu.v_adj_faces = self.v_adj_faces.to(device)
            adjacency_gpu.v_adj_faces_offsets = self.v_adj_faces_offsets.to(device)

            adjacency_gpu.v_adj_edges = self.v_adj_edges.to(device)
            adjacency_gpu.v_adj_edges_offsets = self.v_adj_edges_offsets.to(device)

            adjacency_gpu.v_adj_springs = self.v_adj_springs.to(device)
            adjacency_gpu.v_adj_springs_offsets = self.v_adj_springs_offsets.to(device)

            return adjacency_gpu


@wp.func
def get_vertex_num_adjacent_edges(adjacency: ParticleForceElementAdjacencyInfo, vertex: wp.int32):
    return (adjacency.v_adj_edges_offsets[vertex + 1] - adjacency.v_adj_edges_offsets[vertex]) >> 1


@wp.func
def get_vertex_adjacent_edge_id_order(adjacency: ParticleForceElementAdjacencyInfo, vertex: wp.int32, edge: wp.int32):
    offset = adjacency.v_adj_edges_offsets[vertex]
    return adjacency.v_adj_edges[offset + edge * 2], adjacency.v_adj_edges[offset + edge * 2 + 1]


@wp.func
def get_vertex_num_adjacent_faces(adjacency: ParticleForceElementAdjacencyInfo, vertex: wp.int32):
    return (adjacency.v_adj_faces_offsets[vertex + 1] - adjacency.v_adj_faces_offsets[vertex]) >> 1


@wp.func
def get_vertex_adjacent_face_id_order(adjacency: ParticleForceElementAdjacencyInfo, vertex: wp.int32, face: wp.int32):
    offset = adjacency.v_adj_faces_offsets[vertex]
    return adjacency.v_adj_faces[offset + face * 2], adjacency.v_adj_faces[offset + face * 2 + 1]


@wp.func
def get_vertex_num_adjacent_springs(adjacency: ParticleForceElementAdjacencyInfo, vertex: wp.int32):
    return adjacency.v_adj_springs_offsets[vertex + 1] - adjacency.v_adj_springs_offsets[vertex]


@wp.func
def get_vertex_adjacent_spring_id(adjacency: ParticleForceElementAdjacencyInfo, vertex: wp.int32, spring: wp.int32):
    offset = adjacency.v_adj_springs_offsets[vertex]
    return adjacency.v_adj_springs[offset + spring]


@wp.kernel
def count_num_adjacent_edges(
    edges_array: wp.array(dtype=wp.int32, ndim=2), num_vertex_adjacent_edges: wp.array(dtype=wp.int32)
):
    for edge_id in range(edges_array.shape[0]):
        o0 = edges_array[edge_id, 0]
        o1 = edges_array[edge_id, 1]

        v0 = edges_array[edge_id, 2]
        v1 = edges_array[edge_id, 3]

        num_vertex_adjacent_edges[v0] = num_vertex_adjacent_edges[v0] + 1
        num_vertex_adjacent_edges[v1] = num_vertex_adjacent_edges[v1] + 1

        if o0 != -1:
            num_vertex_adjacent_edges[o0] = num_vertex_adjacent_edges[o0] + 1
        if o1 != -1:
            num_vertex_adjacent_edges[o1] = num_vertex_adjacent_edges[o1] + 1


@wp.kernel
def fill_adjacent_edges(
    edges_array: wp.array(dtype=wp.int32, ndim=2),
    vertex_adjacent_edges_offsets: wp.array(dtype=wp.int32),
    vertex_adjacent_edges_fill_count: wp.array(dtype=wp.int32),
    vertex_adjacent_edges: wp.array(dtype=wp.int32),
):
    for edge_id in range(edges_array.shape[0]):
        v0 = edges_array[edge_id, 2]
        v1 = edges_array[edge_id, 3]

        fill_count_v0 = vertex_adjacent_edges_fill_count[v0]
        buffer_offset_v0 = vertex_adjacent_edges_offsets[v0]
        vertex_adjacent_edges[buffer_offset_v0 + fill_count_v0 * 2] = edge_id
        vertex_adjacent_edges[buffer_offset_v0 + fill_count_v0 * 2 + 1] = 2
        vertex_adjacent_edges_fill_count[v0] = fill_count_v0 + 1

        fill_count_v1 = vertex_adjacent_edges_fill_count[v1]
        buffer_offset_v1 = vertex_adjacent_edges_offsets[v1]
        vertex_adjacent_edges[buffer_offset_v1 + fill_count_v1 * 2] = edge_id
        vertex_adjacent_edges[buffer_offset_v1 + fill_count_v1 * 2 + 1] = 3
        vertex_adjacent_edges_fill_count[v1] = fill_count_v1 + 1

        o0 = edges_array[edge_id, 0]
        if o0 != -1:
            fill_count_o0 = vertex_adjacent_edges_fill_count[o0]
            buffer_offset_o0 = vertex_adjacent_edges_offsets[o0]
            vertex_adjacent_edges[buffer_offset_o0 + fill_count_o0 * 2] = edge_id
            vertex_adjacent_edges[buffer_offset_o0 + fill_count_o0 * 2 + 1] = 0
            vertex_adjacent_edges_fill_count[o0] = fill_count_o0 + 1

        o1 = edges_array[edge_id, 1]
        if o1 != -1:
            fill_count_o1 = vertex_adjacent_edges_fill_count[o1]
            buffer_offset_o1 = vertex_adjacent_edges_offsets[o1]
            vertex_adjacent_edges[buffer_offset_o1 + fill_count_o1 * 2] = edge_id
            vertex_adjacent_edges[buffer_offset_o1 + fill_count_o1 * 2 + 1] = 1
            vertex_adjacent_edges_fill_count[o1] = fill_count_o1 + 1


@wp.kernel
def count_num_adjacent_faces(
    face_indices: wp.array(dtype=wp.int32, ndim=2), num_vertex_adjacent_faces: wp.array(dtype=wp.int32)
):
    for face in range(face_indices.shape[0]):
        v0 = face_indices[face, 0]
        v1 = face_indices[face, 1]
        v2 = face_indices[face, 2]

        num_vertex_adjacent_faces[v0] = num_vertex_adjacent_faces[v0] + 1
        num_vertex_adjacent_faces[v1] = num_vertex_adjacent_faces[v1] + 1
        num_vertex_adjacent_faces[v2] = num_vertex_adjacent_faces[v2] + 1


@wp.kernel
def fill_adjacent_faces(
    face_indices: wp.array(dtype=wp.int32, ndim=2),
    vertex_adjacent_faces_offsets: wp.array(dtype=wp.int32),
    vertex_adjacent_faces_fill_count: wp.array(dtype=wp.int32),
    vertex_adjacent_faces: wp.array(dtype=wp.int32),
):
    for face in range(face_indices.shape[0]):
        v0 = face_indices[face, 0]
        v1 = face_indices[face, 1]
        v2 = face_indices[face, 2]

        fill_count_v0 = vertex_adjacent_faces_fill_count[v0]
        buffer_offset_v0 = vertex_adjacent_faces_offsets[v0]
        vertex_adjacent_faces[buffer_offset_v0 + fill_count_v0 * 2] = face
        vertex_adjacent_faces[buffer_offset_v0 + fill_count_v0 * 2 + 1] = 0
        vertex_adjacent_faces_fill_count[v0] = fill_count_v0 + 1

        fill_count_v1 = vertex_adjacent_faces_fill_count[v1]
        buffer_offset_v1 = vertex_adjacent_faces_offsets[v1]
        vertex_adjacent_faces[buffer_offset_v1 + fill_count_v1 * 2] = face
        vertex_adjacent_faces[buffer_offset_v1 + fill_count_v1 * 2 + 1] = 1
        vertex_adjacent_faces_fill_count[v1] = fill_count_v1 + 1

        fill_count_v2 = vertex_adjacent_faces_fill_count[v2]
        buffer_offset_v2 = vertex_adjacent_faces_offsets[v2]
        vertex_adjacent_faces[buffer_offset_v2 + fill_count_v2 * 2] = face
        vertex_adjacent_faces[buffer_offset_v2 + fill_count_v2 * 2 + 1] = 2
        vertex_adjacent_faces_fill_count[v2] = fill_count_v2 + 1


@wp.kernel
def count_num_adjacent_springs(
    springs_array: wp.array(dtype=wp.int32), num_vertex_adjacent_springs: wp.array(dtype=wp.int32)
):
    num_springs = springs_array.shape[0] / 2
    for spring_id in range(num_springs):
        v0 = springs_array[spring_id * 2]
        v1 = springs_array[spring_id * 2 + 1]

        num_vertex_adjacent_springs[v0] = num_vertex_adjacent_springs[v0] + 1
        num_vertex_adjacent_springs[v1] = num_vertex_adjacent_springs[v1] + 1


@wp.kernel
def fill_adjacent_springs(
    springs_array: wp.array(dtype=wp.int32),
    vertex_adjacent_springs_offsets: wp.array(dtype=wp.int32),
    vertex_adjacent_springs_fill_count: wp.array(dtype=wp.int32),
    vertex_adjacent_springs: wp.array(dtype=wp.int32),
):
    num_springs = springs_array.shape[0] / 2
    for spring_id in range(num_springs):
        v0 = springs_array[spring_id * 2]
        v1 = springs_array[spring_id * 2 + 1]

        fill_count_v0 = vertex_adjacent_springs_fill_count[v0]
        buffer_offset_v0 = vertex_adjacent_springs_offsets[v0]
        vertex_adjacent_springs[buffer_offset_v0 + fill_count_v0] = spring_id
        vertex_adjacent_springs_fill_count[v0] = fill_count_v0 + 1

        fill_count_v1 = vertex_adjacent_springs_fill_count[v1]
        buffer_offset_v1 = vertex_adjacent_springs_offsets[v1]
        vertex_adjacent_springs[buffer_offset_v1 + fill_count_v1] = spring_id
        vertex_adjacent_springs_fill_count[v1] = fill_count_v1 + 1


@wp.kernel
def _test_compute_force_element_adjacency(
    adjacency: ParticleForceElementAdjacencyInfo,
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    face_indices: wp.array(dtype=wp.int32, ndim=2),
):
    wp.printf("num vertices: %d\n", adjacency.v_adj_edges_offsets.shape[0] - 1)
    for vertex in range(adjacency.v_adj_edges_offsets.shape[0] - 1):
        num_adj_edges = get_vertex_num_adjacent_edges(adjacency, vertex)
        for i_bd in range(num_adj_edges):
            bd_id, v_order = get_vertex_adjacent_edge_id_order(adjacency, vertex, i_bd)

            if edge_indices[bd_id, v_order] != vertex:
                print("Error!!!")
                wp.printf("vertex: %d | num_adj_edges: %d\n", vertex, num_adj_edges)
                wp.printf("--iBd: %d | ", i_bd)
                wp.printf("edge id: %d | v_order: %d\n", bd_id, v_order)

        num_adj_faces = get_vertex_num_adjacent_faces(adjacency, vertex)

        for i_face in range(num_adj_faces):
            face, v_order = get_vertex_adjacent_face_id_order(
                adjacency,
                vertex,
                i_face,
            )

            if face_indices[face, v_order] != vertex:
                print("Error!!!")
                wp.printf("vertex: %d | num_adj_faces: %d\n", vertex, num_adj_faces)
                wp.printf("--i_face: %d | face id: %d | v_order: %d\n", i_face, face, v_order)
                wp.printf(
                    "--face: %d %d %d\n",
                    face_indices[face, 0],
                    face_indices[face, 1],
                    face_indices[face, 2],
                )


@wp.func
def build_orthonormal_basis(n: wp.vec3):
    """
    Builds an orthonormal basis given a normal vector `n`. Return the two axes that is perpendicular to `n`.

    :param n: A 3D vector (list or array-like) representing the normal vector
    """
    b1 = wp.vec3()
    b2 = wp.vec3()
    if n[2] < 0.0:
        a = 1.0 / (1.0 - n[2])
        b = n[0] * n[1] * a
        b1[0] = 1.0 - n[0] * n[0] * a
        b1[1] = -b
        b1[2] = n[0]

        b2[0] = b
        b2[1] = n[1] * n[1] * a - 1.0
        b2[2] = -n[1]
    else:
        a = 1.0 / (1.0 + n[2])
        b = -n[0] * n[1] * a
        b1[0] = 1.0 - n[0] * n[0] * a
        b1[1] = b
        b1[2] = -n[0]

        b2[0] = b
        b2[1] = 1.0 - n[1] * n[1] * a
        b2[2] = -n[1]

    return b1, b2


@wp.func
def evaluate_stvk_force_hessian(
    face: int,
    v_order: int,
    pos: wp.array(dtype=wp.vec3),
    pos_anchor: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    tri_pose: wp.mat22,
    area: float,
    mu: float,
    lmbd: float,
    damping: float,
    dt: float,
):
    # StVK energy density: psi = mu * ||G||_F^2 + 0.5 * lambda * (trace(G))^2

    # Deformation gradient F = [f0, f1] (3x2 matrix as two 3D column vectors)
    v0 = tri_indices[face, 0]
    v1 = tri_indices[face, 1]
    v2 = tri_indices[face, 2]

    x0 = pos[v0]
    x01 = pos[v1] - x0
    x02 = pos[v2] - x0

    # Cache tri_pose elements
    DmInv00 = tri_pose[0, 0]
    DmInv01 = tri_pose[0, 1]
    DmInv10 = tri_pose[1, 0]
    DmInv11 = tri_pose[1, 1]

    # Compute F columns directly: F = [x01, x02] * tri_pose = [f0, f1]
    f0 = x01 * DmInv00 + x02 * DmInv10
    f1 = x01 * DmInv01 + x02 * DmInv11

    # Green strain tensor: G = 0.5(F^T F - I) = [[G00, G01], [G01, G11]] (symmetric 2x2)
    f0_dot_f0 = wp.dot(f0, f0)
    f1_dot_f1 = wp.dot(f1, f1)
    f0_dot_f1 = wp.dot(f0, f1)

    G00 = 0.5 * (f0_dot_f0 - 1.0)
    G11 = 0.5 * (f1_dot_f1 - 1.0)
    G01 = 0.5 * f0_dot_f1

    # Frobenius norm squared of Green strain: ||G||_F^2 = G00^2 + G11^2 + 2 * G01^2
    G_frobenius_sq = G00 * G00 + G11 * G11 + 2.0 * G01 * G01
    if G_frobenius_sq < 1.0e-20:
        return wp.vec3(0.0), wp.mat33(0.0)

    trace_G = G00 + G11

    # First Piola-Kirchhoff stress tensor (StVK model)
    # PK1 = 2*mu*F*G + lambda*trace(G)*F = [PK1_col0, PK1_col1] (3x2)
    lambda_trace_G = lmbd * trace_G
    two_mu = 2.0 * mu

    PK1_col0 = f0 * (two_mu * G00 + lambda_trace_G) + f1 * (two_mu * G01)
    PK1_col1 = f0 * (two_mu * G01) + f1 * (two_mu * G11 + lambda_trace_G)

    # Vertex selection using masks to avoid branching
    mask0 = float(v_order == 0)
    mask1 = float(v_order == 1)
    mask2 = float(v_order == 2)

    # Deformation gradient derivatives w.r.t. current vertex position
    df0_dx = DmInv00 * (mask1 - mask0) + DmInv10 * (mask2 - mask0)
    df1_dx = DmInv01 * (mask1 - mask0) + DmInv11 * (mask2 - mask0)

    # Force via chain rule: force = -(dpsi/dF) : (dF/dx)
    dpsi_dx = PK1_col0 * df0_dx + PK1_col1 * df1_dx
    force = -dpsi_dx

    # Hessian computation using Cauchy-Green invariants
    df0_dx_sq = df0_dx * df0_dx
    df1_dx_sq = df1_dx * df1_dx
    df0_df1_cross = df0_dx * df1_dx

    Ic = f0_dot_f0 + f1_dot_f1
    two_dpsi_dIc = -mu + (0.5 * Ic - 1.0) * lmbd
    I33 = wp.identity(n=3, dtype=float)

    f0_outer_f0 = wp.outer(f0, f0)
    f1_outer_f1 = wp.outer(f1, f1)
    f0_outer_f1 = wp.outer(f0, f1)
    f1_outer_f0 = wp.outer(f1, f0)

    H_IIc00_scaled = mu * (f0_dot_f0 * I33 + 2.0 * f0_outer_f0 + f1_outer_f1)
    H_IIc11_scaled = mu * (f1_dot_f1 * I33 + 2.0 * f1_outer_f1 + f0_outer_f0)
    H_IIc01_scaled = mu * (f0_dot_f1 * I33 + f1_outer_f0)

    # d2(psi)/dF^2 components
    d2E_dF2_00 = lmbd * f0_outer_f0 + two_dpsi_dIc * I33 + H_IIc00_scaled
    d2E_dF2_01 = lmbd * f0_outer_f1 + H_IIc01_scaled
    d2E_dF2_11 = lmbd * f1_outer_f1 + two_dpsi_dIc * I33 + H_IIc11_scaled

    # Chain rule: H = (dF/dx)^T * (d2(psi)/dF^2) * (dF/dx)
    hessian = df0_dx_sq * d2E_dF2_00 + df1_dx_sq * d2E_dF2_11 + df0_df1_cross * (d2E_dF2_01 + wp.transpose(d2E_dF2_01))

    if damping > 0.0:
        inv_dt = 1.0 / dt

        # Previous deformation gradient for velocity
        x0_prev = pos_anchor[v0]
        x01_prev = pos_anchor[v1] - x0_prev
        x02_prev = pos_anchor[v2] - x0_prev

        vel_x01 = (x01 - x01_prev) * inv_dt
        vel_x02 = (x02 - x02_prev) * inv_dt

        df0_dt = vel_x01 * DmInv00 + vel_x02 * DmInv10
        df1_dt = vel_x01 * DmInv01 + vel_x02 * DmInv11

        # First constraint: Cmu = ||G||_F (Frobenius norm of Green strain)
        Cmu = wp.sqrt(G_frobenius_sq)

        G00_normalized = G00 / Cmu
        G01_normalized = G01 / Cmu
        G11_normalized = G11 / Cmu

        # Time derivative of Green strain: dG/dt = 0.5 * (F^T * dF/dt + (dF/dt)^T * F)
        dG_dt_00 = wp.dot(f0, df0_dt)  # dG00/dt
        dG_dt_11 = wp.dot(f1, df1_dt)  # dG11/dt
        dG_dt_01 = 0.5 * (wp.dot(f0, df1_dt) + wp.dot(f1, df0_dt))  # dG01/dt

        # Time derivative of first constraint: dCmu/dt = (1/||G||_F) * (G : dG/dt)
        dCmu_dt = G00_normalized * dG_dt_00 + G11_normalized * dG_dt_11 + 2.0 * G01_normalized * dG_dt_01

        # Gradient of first constraint w.r.t. deformation gradient: dCmu/dF = (G/||G||_F) * F
        dCmu_dF_col0 = G00_normalized * f0 + G01_normalized * f1  # dCmu/df0
        dCmu_dF_col1 = G01_normalized * f0 + G11_normalized * f1  # dCmu/df1

        # Gradient of constraint w.r.t. vertex position: dCmu/dx = (dCmu/dF) : (dF/dx)
        dCmu_dx = df0_dx * dCmu_dF_col0 + df1_dx * dCmu_dF_col1

        # Damping force from first constraint: -mu * damping * (dCmu/dt) * (dCmu/dx)
        kd_mu = mu * damping
        force += -kd_mu * dCmu_dt * dCmu_dx

        # Damping Hessian: mu * damping * (1/dt) * (dCmu/dx) x (dCmu/dx)
        hessian += kd_mu * inv_dt * wp.outer(dCmu_dx, dCmu_dx)

        # Second constraint: Clmbd = trace(G) = G00 + G11 (trace of Green strain)
        # Time derivative of second constraint: dClmbd/dt = trace(dG/dt)
        dClmbd_dt = dG_dt_00 + dG_dt_11

        # Gradient of second constraint w.r.t. deformation gradient: dClmbd/dF = F
        dClmbd_dF_col0 = f0  # dClmbd/df0
        dClmbd_dF_col1 = f1  # dClmbd/df1

        # Gradient of Clmbd w.r.t. vertex position: dClmbd/dx = (dClmbd/dF) : (dF/dx)
        dClmbd_dx = df0_dx * dClmbd_dF_col0 + df1_dx * dClmbd_dF_col1

        # Damping force from second constraint: -lambda * damping * (dClmbd/dt) * (dClmbd/dx)
        kd_lmbd = lmbd * damping
        force += -kd_lmbd * dClmbd_dt * dClmbd_dx

        # Damping Hessian from second constraint: lambda * damping * (1/dt) * (dClmbd/dx) x (dClmbd/dx)
        hessian += kd_lmbd * inv_dt * wp.outer(dClmbd_dx, dClmbd_dx)

    # Apply area scaling
    force *= area
    hessian *= area

    return force, hessian


@wp.func
def compute_normalized_vector_derivative(
    unnormalized_vec_length: float, normalized_vec: wp.vec3, unnormalized_vec_derivative: wp.mat33
) -> wp.mat33:
    projection_matrix = wp.identity(n=3, dtype=float) - wp.outer(normalized_vec, normalized_vec)

    # d(normalized_vec)/dx = (1/|unnormalized_vec|) * (I - normalized_vec * normalized_vec^T) * d(unnormalized_vec)/dx
    return (1.0 / unnormalized_vec_length) * projection_matrix * unnormalized_vec_derivative


@wp.func
def compute_angle_derivative(
    n1_hat: wp.vec3,
    n2_hat: wp.vec3,
    e_hat: wp.vec3,
    dn1hat_dx: wp.mat33,
    dn2hat_dx: wp.mat33,
    sin_theta: float,
    cos_theta: float,
    skew_n1: wp.mat33,
    skew_n2: wp.mat33,
) -> wp.vec3:
    dsin_dx = wp.transpose(skew_n1 * dn2hat_dx - skew_n2 * dn1hat_dx) * e_hat
    dcos_dx = wp.transpose(dn1hat_dx) * n2_hat + wp.transpose(dn2hat_dx) * n1_hat

    # dtheta/dx = dsin/dx * cos - dcos/dx * sin
    return dsin_dx * cos_theta - dcos_dx * sin_theta


@wp.func
def evaluate_dihedral_angle_based_bending_force_hessian(
    bending_index: int,
    v_order: int,
    pos: wp.array(dtype=wp.vec3),
    pos_anchor: wp.array(dtype=wp.vec3),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_rest_angle: wp.array(dtype=float),
    edge_rest_length: wp.array(dtype=float),
    stiffness: float,
    damping: float,
    dt: float,
):
    # Skip invalid edges (boundary edges with missing opposite vertices)
    if edge_indices[bending_index, 0] == -1 or edge_indices[bending_index, 1] == -1:
        return wp.vec3(0.0), wp.mat33(0.0)

    eps = 1.0e-6

    vi0 = edge_indices[bending_index, 0]
    vi1 = edge_indices[bending_index, 1]
    vi2 = edge_indices[bending_index, 2]
    vi3 = edge_indices[bending_index, 3]

    x0 = pos[vi0]  # opposite 0
    x1 = pos[vi1]  # opposite 1
    x2 = pos[vi2]  # edge start
    x3 = pos[vi3]  # edge end

    # Compute edge vectors
    x02 = x2 - x0
    x03 = x3 - x0
    x13 = x3 - x1
    x12 = x2 - x1
    e = x3 - x2

    # Compute normals
    n1 = wp.cross(x02, x03)
    n2 = wp.cross(x13, x12)

    n1_norm = wp.length(n1)
    n2_norm = wp.length(n2)
    e_norm = wp.length(e)

    # Early exit for degenerate cases
    if n1_norm < eps or n2_norm < eps or e_norm < eps:
        return wp.vec3(0.0), wp.mat33(0.0)

    n1_hat = n1 / n1_norm
    n2_hat = n2 / n2_norm
    e_hat = e / e_norm

    sin_theta = wp.dot(wp.cross(n1_hat, n2_hat), e_hat)
    cos_theta = wp.dot(n1_hat, n2_hat)
    theta = wp.atan2(sin_theta, cos_theta)

    k = stiffness * edge_rest_length[bending_index]
    dE_dtheta = k * (theta - edge_rest_angle[bending_index])

    # Pre-compute skew matrices (shared across all angle derivative computations)
    skew_e = wp.skew(e)
    skew_x03 = wp.skew(x03)
    skew_x02 = wp.skew(x02)
    skew_x13 = wp.skew(x13)
    skew_x12 = wp.skew(x12)
    skew_n1 = wp.skew(n1_hat)
    skew_n2 = wp.skew(n2_hat)

    # Compute the derivatives of unit normals with respect to each vertex; required for computing angle derivatives
    dn1hat_dx0 = compute_normalized_vector_derivative(n1_norm, n1_hat, skew_e)
    dn2hat_dx0 = wp.mat33(0.0)

    dn1hat_dx1 = wp.mat33(0.0)
    dn2hat_dx1 = compute_normalized_vector_derivative(n2_norm, n2_hat, -skew_e)

    dn1hat_dx2 = compute_normalized_vector_derivative(n1_norm, n1_hat, -skew_x03)
    dn2hat_dx2 = compute_normalized_vector_derivative(n2_norm, n2_hat, skew_x13)

    dn1hat_dx3 = compute_normalized_vector_derivative(n1_norm, n1_hat, skew_x02)
    dn2hat_dx3 = compute_normalized_vector_derivative(n2_norm, n2_hat, -skew_x12)

    # Compute all angle derivatives (required for damping)
    dtheta_dx0 = compute_angle_derivative(
        n1_hat, n2_hat, e_hat, dn1hat_dx0, dn2hat_dx0, sin_theta, cos_theta, skew_n1, skew_n2
    )
    dtheta_dx1 = compute_angle_derivative(
        n1_hat, n2_hat, e_hat, dn1hat_dx1, dn2hat_dx1, sin_theta, cos_theta, skew_n1, skew_n2
    )
    dtheta_dx2 = compute_angle_derivative(
        n1_hat, n2_hat, e_hat, dn1hat_dx2, dn2hat_dx2, sin_theta, cos_theta, skew_n1, skew_n2
    )
    dtheta_dx3 = compute_angle_derivative(
        n1_hat, n2_hat, e_hat, dn1hat_dx3, dn2hat_dx3, sin_theta, cos_theta, skew_n1, skew_n2
    )

    # Use float masks for branch-free selection
    mask0 = float(v_order == 0)
    mask1 = float(v_order == 1)
    mask2 = float(v_order == 2)
    mask3 = float(v_order == 3)

    # Select the derivative for the current vertex without branching
    dtheta_dx = dtheta_dx0 * mask0 + dtheta_dx1 * mask1 + dtheta_dx2 * mask2 + dtheta_dx3 * mask3

    # Compute elastic force and hessian
    bending_force = -dE_dtheta * dtheta_dx
    bending_hessian = k * wp.outer(dtheta_dx, dtheta_dx)

    if damping > 0.0:
        inv_dt = 1.0 / dt
        x_prev0 = pos_anchor[vi0]
        x_prev1 = pos_anchor[vi1]
        x_prev2 = pos_anchor[vi2]
        x_prev3 = pos_anchor[vi3]

        # Compute displacement vectors
        dx0 = x0 - x_prev0
        dx1 = x1 - x_prev1
        dx2 = x2 - x_prev2
        dx3 = x3 - x_prev3

        # Compute angular velocity using all derivatives
        dtheta_dt = (
            wp.dot(dtheta_dx0, dx0) + wp.dot(dtheta_dx1, dx1) + wp.dot(dtheta_dx2, dx2) + wp.dot(dtheta_dx3, dx3)
        ) * inv_dt

        damping_coeff = damping * k  # damping coefficients following the VBD convention
        damping_force = -damping_coeff * dtheta_dt * dtheta_dx
        damping_hessian = damping_coeff * inv_dt * wp.outer(dtheta_dx, dtheta_dx)

        bending_force = bending_force + damping_force
        bending_hessian = bending_hessian + damping_hessian

    return bending_force, bending_hessian


@wp.func
def evaluate_self_contact_force_norm(dis: float, collision_radius: float, k: float):
    # Adjust distance and calculate penetration depth

    penetration_depth = collision_radius - dis

    # Initialize outputs
    dEdD = wp.float32(0.0)
    d2E_dDdD = wp.float32(0.0)

    # C2 continuity calculation
    tau = collision_radius * 0.5
    if tau > dis > 1e-5:
        k2 = 0.5 * tau * tau * k
        dEdD = -k2 / dis
        d2E_dDdD = k2 / (dis * dis)
    else:
        dEdD = -k * penetration_depth
        d2E_dDdD = k

    return dEdD, d2E_dDdD


@wp.func
def damp_collision(
    displacement: wp.vec3,
    collision_normal: wp.vec3,
    collision_hessian: wp.mat33,
    collision_damping: float,
    dt: float,
):
    if wp.dot(displacement, collision_normal) > 0:
        damping_hessian = (collision_damping / dt) * collision_hessian
        damping_force = damping_hessian * displacement
        return damping_force, damping_hessian
    else:
        return wp.vec3(0.0), wp.mat33(0.0)


@wp.func
def evaluate_edge_edge_contact(
    v: int,
    v_order: int,
    e1: int,
    e2: int,
    pos: wp.array(dtype=wp.vec3),
    pos_anchor: wp.array(dtype=wp.vec3),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    collision_radius: float,
    collision_stiffness: float,
    collision_damping: float,
    friction_coefficient: float,
    friction_epsilon: float,
    dt: float,
    edge_edge_parallel_epsilon: float,
):
    r"""
    Returns the edge-edge contact force and hessian, including the friction force.
    Args:
        v:
        v_order: \in {0, 1, 2, 3}, 0, 1 is vertex 0, 1 of e1, 2,3 is vertex 0, 1 of e2
        e0
        e1
        pos
        pos_anchor,
        edge_indices
        collision_radius
        collision_stiffness
        dt
        edge_edge_parallel_epsilon: threshold to determine whether 2 edges are parallel
    """
    e1_v1 = edge_indices[e1, 2]
    e1_v2 = edge_indices[e1, 3]

    e1_v1_pos = pos[e1_v1]
    e1_v2_pos = pos[e1_v2]

    e2_v1 = edge_indices[e2, 2]
    e2_v2 = edge_indices[e2, 3]

    e2_v1_pos = pos[e2_v1]
    e2_v2_pos = pos[e2_v2]

    st = wp.closest_point_edge_edge(e1_v1_pos, e1_v2_pos, e2_v1_pos, e2_v2_pos, edge_edge_parallel_epsilon)
    s = st[0]
    t = st[1]
    e1_vec = e1_v2_pos - e1_v1_pos
    e2_vec = e2_v2_pos - e2_v1_pos
    c1 = e1_v1_pos + e1_vec * s
    c2 = e2_v1_pos + e2_vec * t

    # c1, c2, s, t = closest_point_edge_edge_2(e1_v1_pos, e1_v2_pos, e2_v1_pos, e2_v2_pos)

    diff = c1 - c2
    dis = st[2]
    collision_normal = diff / dis

    if dis < collision_radius:
        bs = wp.vec4(1.0 - s, s, -1.0 + t, -t)
        v_bary = bs[v_order]

        dEdD, d2E_dDdD = evaluate_self_contact_force_norm(dis, collision_radius, collision_stiffness)

        collision_force = -dEdD * v_bary * collision_normal
        collision_hessian = d2E_dDdD * v_bary * v_bary * wp.outer(collision_normal, collision_normal)

        # friction
        c1_prev = pos_anchor[e1_v1] + (pos_anchor[e1_v2] - pos_anchor[e1_v1]) * s
        c2_prev = pos_anchor[e2_v1] + (pos_anchor[e2_v2] - pos_anchor[e2_v1]) * t

        dx = (c1 - c1_prev) - (c2 - c2_prev)
        axis_1, axis_2 = build_orthonormal_basis(collision_normal)

        T = mat32(
            axis_1[0],
            axis_2[0],
            axis_1[1],
            axis_2[1],
            axis_1[2],
            axis_2[2],
        )

        u = wp.transpose(T) * dx
        eps_U = friction_epsilon * dt

        # fmt: off
        if wp.static("contact_force_hessian_ee" in VBD_DEBUG_PRINTING_OPTIONS):
            wp.printf(
                "    collision force:\n    %f %f %f,\n    collision hessian:\n    %f %f %f,\n    %f %f %f,\n    %f %f %f\n",
                collision_force[0], collision_force[1], collision_force[2], collision_hessian[0, 0], collision_hessian[0, 1], collision_hessian[0, 2], collision_hessian[1, 0], collision_hessian[1, 1], collision_hessian[1, 2], collision_hessian[2, 0], collision_hessian[2, 1], collision_hessian[2, 2],
            )
        # fmt: on

        friction_force, friction_hessian = compute_friction(friction_coefficient, -dEdD, T, u, eps_U)
        friction_force = friction_force * v_bary
        friction_hessian = friction_hessian * v_bary * v_bary

        # # fmt: off
        # if wp.static("contact_force_hessian_ee" in VBD_DEBUG_PRINTING_OPTIONS):
        #     wp.printf(
        #         "    friction force:\n    %f %f %f,\n    friction hessian:\n    %f %f %f,\n    %f %f %f,\n    %f %f %f\n",
        #         friction_force[0], friction_force[1], friction_force[2], friction_hessian[0, 0], friction_hessian[0, 1], friction_hessian[0, 2], friction_hessian[1, 0], friction_hessian[1, 1], friction_hessian[1, 2], friction_hessian[2, 0], friction_hessian[2, 1], friction_hessian[2, 2],
        #     )
        # # fmt: on

        if v_order == 0:
            displacement = pos_anchor[e1_v1] - e1_v1_pos
        elif v_order == 1:
            displacement = pos_anchor[e1_v2] - e1_v2_pos
        elif v_order == 2:
            displacement = pos_anchor[e2_v1] - e2_v1_pos
        else:
            displacement = pos_anchor[e2_v2] - e2_v2_pos

        collision_normal_sign = wp.vec4(1.0, 1.0, -1.0, -1.0)
        if wp.dot(displacement, collision_normal * collision_normal_sign[v_order]) > 0:
            damping_hessian = (collision_damping / dt) * collision_hessian
            collision_hessian = collision_hessian + damping_hessian
            collision_force = collision_force + damping_hessian * displacement

        collision_force = collision_force + friction_force
        collision_hessian = collision_hessian + friction_hessian
    else:
        collision_force = wp.vec3(0.0, 0.0, 0.0)
        collision_hessian = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    return collision_force, collision_hessian


@wp.func
def evaluate_edge_edge_contact_2_vertices(
    e1: int,
    e2: int,
    pos: wp.array(dtype=wp.vec3),
    pos_anchor: wp.array(dtype=wp.vec3),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    collision_radius: float,
    collision_stiffness: float,
    collision_damping: float,
    friction_coefficient: float,
    friction_epsilon: float,
    dt: float,
    edge_edge_parallel_epsilon: float,
):
    r"""
    Returns the edge-edge contact force and hessian, including the friction force.
    Args:
        v:
        v_order: \in {0, 1, 2, 3}, 0, 1 is vertex 0, 1 of e1, 2,3 is vertex 0, 1 of e2
        e0
        e1
        pos
        edge_indices
        collision_radius
        collision_stiffness
        dt
    """
    e1_v1 = edge_indices[e1, 2]
    e1_v2 = edge_indices[e1, 3]

    e1_v1_pos = pos[e1_v1]
    e1_v2_pos = pos[e1_v2]

    e2_v1 = edge_indices[e2, 2]
    e2_v2 = edge_indices[e2, 3]

    e2_v1_pos = pos[e2_v1]
    e2_v2_pos = pos[e2_v2]

    st = wp.closest_point_edge_edge(e1_v1_pos, e1_v2_pos, e2_v1_pos, e2_v2_pos, edge_edge_parallel_epsilon)
    s = st[0]
    t = st[1]
    e1_vec = e1_v2_pos - e1_v1_pos
    e2_vec = e2_v2_pos - e2_v1_pos
    c1 = e1_v1_pos + e1_vec * s
    c2 = e2_v1_pos + e2_vec * t

    # c1, c2, s, t = closest_point_edge_edge_2(e1_v1_pos, e1_v2_pos, e2_v1_pos, e2_v2_pos)

    diff = c1 - c2
    dis = st[2]
    collision_normal = diff / dis

    if 0.0 < dis < collision_radius:
        bs = wp.vec4(1.0 - s, s, -1.0 + t, -t)

        dEdD, d2E_dDdD = evaluate_self_contact_force_norm(dis, collision_radius, collision_stiffness)

        collision_force = -dEdD * collision_normal
        collision_hessian = d2E_dDdD * wp.outer(collision_normal, collision_normal)

        # friction
        c1_prev = pos_anchor[e1_v1] + (pos_anchor[e1_v2] - pos_anchor[e1_v1]) * s
        c2_prev = pos_anchor[e2_v1] + (pos_anchor[e2_v2] - pos_anchor[e2_v1]) * t

        dx = (c1 - c1_prev) - (c2 - c2_prev)
        axis_1, axis_2 = build_orthonormal_basis(collision_normal)

        T = mat32(
            axis_1[0],
            axis_2[0],
            axis_1[1],
            axis_2[1],
            axis_1[2],
            axis_2[2],
        )

        u = wp.transpose(T) * dx
        eps_U = friction_epsilon * dt

        # fmt: off
        if wp.static("contact_force_hessian_ee" in VBD_DEBUG_PRINTING_OPTIONS):
            wp.printf(
                "    collision force:\n    %f %f %f,\n    collision hessian:\n    %f %f %f,\n    %f %f %f,\n    %f %f %f\n",
                collision_force[0], collision_force[1], collision_force[2], collision_hessian[0, 0], collision_hessian[0, 1], collision_hessian[0, 2], collision_hessian[1, 0], collision_hessian[1, 1], collision_hessian[1, 2], collision_hessian[2, 0], collision_hessian[2, 1], collision_hessian[2, 2],
            )
        # fmt: on

        friction_force, friction_hessian = compute_friction(friction_coefficient, -dEdD, T, u, eps_U)

        # # fmt: off
        # if wp.static("contact_force_hessian_ee" in VBD_DEBUG_PRINTING_OPTIONS):
        #     wp.printf(
        #         "    friction force:\n    %f %f %f,\n    friction hessian:\n    %f %f %f,\n    %f %f %f,\n    %f %f %f\n",
        #         friction_force[0], friction_force[1], friction_force[2], friction_hessian[0, 0], friction_hessian[0, 1], friction_hessian[0, 2], friction_hessian[1, 0], friction_hessian[1, 1], friction_hessian[1, 2], friction_hessian[2, 0], friction_hessian[2, 1], friction_hessian[2, 2],
        #     )
        # # fmt: on

        displacement_0 = pos_anchor[e1_v1] - e1_v1_pos
        displacement_1 = pos_anchor[e1_v2] - e1_v2_pos

        collision_force_0 = collision_force * bs[0]
        collision_force_1 = collision_force * bs[1]

        collision_hessian_0 = collision_hessian * bs[0] * bs[0]
        collision_hessian_1 = collision_hessian * bs[1] * bs[1]

        collision_normal_sign = wp.vec4(1.0, 1.0, -1.0, -1.0)
        damping_force, damping_hessian = damp_collision(
            displacement_0,
            collision_normal * collision_normal_sign[0],
            collision_hessian_0,
            collision_damping,
            dt,
        )

        collision_force_0 += damping_force + bs[0] * friction_force
        collision_hessian_0 += damping_hessian + bs[0] * bs[0] * friction_hessian

        damping_force, damping_hessian = damp_collision(
            displacement_1,
            collision_normal * collision_normal_sign[1],
            collision_hessian_1,
            collision_damping,
            dt,
        )
        collision_force_1 += damping_force + bs[1] * friction_force
        collision_hessian_1 += damping_hessian + bs[1] * bs[1] * friction_hessian

        return True, collision_force_0, collision_force_1, collision_hessian_0, collision_hessian_1
    else:
        collision_force = wp.vec3(0.0, 0.0, 0.0)
        collision_hessian = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        return False, collision_force, collision_force, collision_hessian, collision_hessian

@wp.func
def evaluate_edge_edge_contact_2_vertices_log_collision(
    e1: int,
    e2: int,
    pos: wp.array(dtype=wp.vec3),
    pos_anchor: wp.array(dtype=wp.vec3),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    collision_radius: float,
    collision_stiffness: float,
    collision_damping: float,
    friction_coefficient: float,
    friction_epsilon: float,
    dt: float,
    edge_edge_parallel_epsilon: float,
):
    r"""
    Returns the edge-edge contact force and hessian, including the friction force.
    Args:
        v:
        v_order: \in {0, 1, 2, 3}, 0, 1 is vertex 0, 1 of e1, 2,3 is vertex 0, 1 of e2
        e0
        e1
        pos
        edge_indices
        collision_radius
        collision_stiffness
        dt
    """
    e1_v1 = edge_indices[e1, 2]
    e1_v2 = edge_indices[e1, 3]

    e1_v1_pos = pos[e1_v1]
    e1_v2_pos = pos[e1_v2]

    e2_v1 = edge_indices[e2, 2]
    e2_v2 = edge_indices[e2, 3]

    e2_v1_pos = pos[e2_v1]
    e2_v2_pos = pos[e2_v2]

    st = wp.closest_point_edge_edge(e1_v1_pos, e1_v2_pos, e2_v1_pos, e2_v2_pos, edge_edge_parallel_epsilon)
    s = st[0]
    t = st[1]
    e1_vec = e1_v2_pos - e1_v1_pos
    e2_vec = e2_v2_pos - e2_v1_pos
    c1 = e1_v1_pos + e1_vec * s
    c2 = e2_v1_pos + e2_vec * t

    # c1, c2, s, t = closest_point_edge_edge_2(e1_v1_pos, e1_v2_pos, e2_v1_pos, e2_v2_pos)

    diff = c1 - c2
    dis = st[2]
    collision_normal = diff / dis

    if 0.0 < dis < collision_radius:
        bs = wp.vec4(1.0 - s, s, -1.0 + t, -t)

        dEdD, d2E_dDdD = evaluate_self_contact_force_norm(dis, collision_radius, collision_stiffness)

        collision_force = -dEdD * collision_normal
        collision_hessian = d2E_dDdD * wp.outer(collision_normal, collision_normal)

        # friction
        c1_prev = pos_anchor[e1_v1] + (pos_anchor[e1_v2] - pos_anchor[e1_v1]) * s
        c2_prev = pos_anchor[e2_v1] + (pos_anchor[e2_v2] - pos_anchor[e2_v1]) * t

        dx = (c1 - c1_prev) - (c2 - c2_prev) # c2 frame 기준 relative c1 disp
        axis_1, axis_2 = build_orthonormal_basis(collision_normal)

        T = mat32(
            axis_1[0],
            axis_2[0],
            axis_1[1],
            axis_2[1],
            axis_1[2],
            axis_2[2],
        )

        u = wp.transpose(T) * dx
        eps_U = friction_epsilon * dt

        # fmt: off
        if wp.static("contact_force_hessian_ee" in VBD_DEBUG_PRINTING_OPTIONS):
            wp.printf(
                "    collision force:\n    %f %f %f,\n    collision hessian:\n    %f %f %f,\n    %f %f %f,\n    %f %f %f\n",
                collision_force[0], collision_force[1], collision_force[2], collision_hessian[0, 0], collision_hessian[0, 1], collision_hessian[0, 2], collision_hessian[1, 0], collision_hessian[1, 1], collision_hessian[1, 2], collision_hessian[2, 0], collision_hessian[2, 1], collision_hessian[2, 2],
            )
        # fmt: on

        friction_force, friction_hessian = compute_friction(friction_coefficient, -dEdD, T, u, eps_U)

        # # fmt: off
        # if wp.static("contact_force_hessian_ee" in VBD_DEBUG_PRINTING_OPTIONS):
        #     wp.printf(
        #         "    friction force:\n    %f %f %f,\n    friction hessian:\n    %f %f %f,\n    %f %f %f,\n    %f %f %f\n",
        #         friction_force[0], friction_force[1], friction_force[2], friction_hessian[0, 0], friction_hessian[0, 1], friction_hessian[0, 2], friction_hessian[1, 0], friction_hessian[1, 1], friction_hessian[1, 2], friction_hessian[2, 0], friction_hessian[2, 1], friction_hessian[2, 2],
        #     )
        # # fmt: on

        displacement_0 = pos_anchor[e1_v1] - e1_v1_pos
        displacement_1 = pos_anchor[e1_v2] - e1_v2_pos

        collision_force_0 = collision_force * bs[0] # e1 v1
        collision_force_1 = collision_force * bs[1] # e1 v2
        contacts_normal_contact_force0 = collision_force_0
        contacts_normal_contact_force1 = collision_force_1

        collision_hessian_0 = collision_hessian * bs[0] * bs[0]
        collision_hessian_1 = collision_hessian * bs[1] * bs[1]

        collision_normal_sign = wp.vec4(1.0, 1.0, -1.0, -1.0)
        damping_force, damping_hessian = damp_collision(
            displacement_0,
            collision_normal * collision_normal_sign[0],
            collision_hessian_0,
            collision_damping,
            dt,
        )
        ncfs = contacts_normal_contact_force0 # normal contact force sum
        ncfm = contacts_normal_contact_force0 # normal contact force min


        collision_force_0 += damping_force + bs[0] * friction_force
        collision_hessian_0 += damping_hessian + bs[0] * bs[0] * friction_hessian

        contacts_friction0 = bs[0] * friction_force

        damping_force, damping_hessian = damp_collision(
            displacement_1,
            collision_normal * collision_normal_sign[1],
            collision_hessian_1,
            collision_damping,
            dt,
        )
        ncfs += contacts_normal_contact_force1
        if wp.length(contacts_normal_contact_force1) < wp.length(ncfm):
            ncfm = contacts_normal_contact_force1

        collision_force_1 += damping_force + bs[1] * friction_force
        collision_hessian_1 += damping_hessian + bs[1] * bs[1] * friction_hessian

        contacts_friction1 = bs[1] * friction_force

        return (True, collision_force_0, collision_force_1, collision_hessian_0, collision_hessian_1,
        
                T,
                collision_normal, 
                wp.length(u), # u_norm
                eps_U,
                friction_force,
                ncfs,
                ncfm,


                contacts_normal_contact_force0,
                contacts_normal_contact_force1,
                wp.vec3(0.0, 0.0, 0.0),
                wp.vec3(0.0, 0.0, 0.0),
                contacts_friction0,
                contacts_friction1,
                wp.vec3(0.0, 0.0, 0.0),
                wp.vec3(0.0, 0.0, 0.0)
        )
    else:
        collision_force = wp.vec3(0.0, 0.0, 0.0)
        collision_hessian = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        T = mat32(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        return (False, collision_force, collision_force, collision_hessian, collision_hessian,
        
                T,
                collision_force, 
                0.0, # u_norm
                0.0,
                collision_force,
                collision_force,
                collision_force,


                collision_force,
                collision_force,
                collision_force,
                collision_force,
                collision_force,
                collision_force,
                collision_force,
                collision_force,
        )


@wp.func
def evaluate_vertex_triangle_collision_force_hessian(
    v: int,
    v_order: int,
    tri: int,
    pos: wp.array(dtype=wp.vec3),
    pos_anchor: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    collision_radius: float,
    collision_stiffness: float,
    collision_damping: float,
    friction_coefficient: float,
    friction_epsilon: float,
    dt: float,
):
    a = pos[tri_indices[tri, 0]]
    b = pos[tri_indices[tri, 1]]
    c = pos[tri_indices[tri, 2]]

    p = pos[v]

    closest_p, bary, _feature_type = triangle_closest_point(a, b, c, p)

    diff = p - closest_p
    dis = wp.length(diff)
    collision_normal = diff / dis

    if dis < collision_radius:
        bs = wp.vec4(-bary[0], -bary[1], -bary[2], 1.0)
        v_bary = bs[v_order]

        dEdD, d2E_dDdD = evaluate_self_contact_force_norm(dis, collision_radius, collision_stiffness)

        collision_force = -dEdD * v_bary * collision_normal
        collision_hessian = d2E_dDdD * v_bary * v_bary * wp.outer(collision_normal, collision_normal)

        # friction force
        dx_v = p - pos_anchor[v]

        closest_p_prev = (
            bary[0] * pos_anchor[tri_indices[tri, 0]]
            + bary[1] * pos_anchor[tri_indices[tri, 1]]
            + bary[2] * pos_anchor[tri_indices[tri, 2]]
        )

        dx = dx_v - (closest_p - closest_p_prev)

        e0, e1 = build_orthonormal_basis(collision_normal)

        T = mat32(e0[0], e1[0], e0[1], e1[1], e0[2], e1[2])

        u = wp.transpose(T) * dx
        eps_U = friction_epsilon * dt

        friction_force, friction_hessian = compute_friction(friction_coefficient, -dEdD, T, u, eps_U)

        # fmt: off
        if wp.static("contact_force_hessian_vt" in VBD_DEBUG_PRINTING_OPTIONS):
            wp.printf(
                "v: %d dEdD: %f\nnormal force: %f %f %f\nfriction force: %f %f %f\n",
                v,
                dEdD,
                collision_force[0], collision_force[1], collision_force[2], friction_force[0], friction_force[1], friction_force[2],
            )
        # fmt: on

        if v_order == 0:
            displacement = pos_anchor[tri_indices[tri, 0]] - a
        elif v_order == 1:
            displacement = pos_anchor[tri_indices[tri, 1]] - b
        elif v_order == 2:
            displacement = pos_anchor[tri_indices[tri, 2]] - c
        else:
            displacement = pos_anchor[v] - p

        collision_normal_sign = wp.vec4(-1.0, -1.0, -1.0, 1.0)
        if wp.dot(displacement, collision_normal * collision_normal_sign[v_order]) > 0:
            damping_hessian = (collision_damping / dt) * collision_hessian
            collision_hessian = collision_hessian + damping_hessian
            collision_force = collision_force + damping_hessian * displacement

        collision_force = collision_force + v_bary * friction_force
        collision_hessian = collision_hessian + v_bary * v_bary * friction_hessian
    else:
        collision_force = wp.vec3(0.0, 0.0, 0.0)
        collision_hessian = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    return collision_force, collision_hessian


@wp.func
def evaluate_vertex_triangle_collision_force_hessian_4_vertices(
    v: int,
    tri: int,
    pos: wp.array(dtype=wp.vec3),
    pos_anchor: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    collision_radius: float,
    collision_stiffness: float,
    collision_damping: float,
    friction_coefficient: float,
    friction_epsilon: float,
    dt: float,
):
    a = pos[tri_indices[tri, 0]]
    b = pos[tri_indices[tri, 1]]
    c = pos[tri_indices[tri, 2]]

    p = pos[v]

    closest_p, bary, _feature_type = triangle_closest_point(a, b, c, p)

    diff = p - closest_p
    dis = wp.length(diff)
    collision_normal = diff / dis

    if 0.0 < dis < collision_radius:
        bs = wp.vec4(-bary[0], -bary[1], -bary[2], 1.0)

        dEdD, d2E_dDdD = evaluate_self_contact_force_norm(dis, collision_radius, collision_stiffness)

        collision_force = -dEdD * collision_normal
        collision_hessian = d2E_dDdD * wp.outer(collision_normal, collision_normal)

        # friction force
        dx_v = p - pos_anchor[v]

        closest_p_prev = (
            bary[0] * pos_anchor[tri_indices[tri, 0]]
            + bary[1] * pos_anchor[tri_indices[tri, 1]]
            + bary[2] * pos_anchor[tri_indices[tri, 2]]
        )

        dx = dx_v - (closest_p - closest_p_prev)

        e0, e1 = build_orthonormal_basis(collision_normal)

        T = mat32(e0[0], e1[0], e0[1], e1[1], e0[2], e1[2])

        u = wp.transpose(T) * dx
        eps_U = friction_epsilon * dt

        friction_force, friction_hessian = compute_friction(friction_coefficient, -dEdD, T, u, eps_U)

        # fmt: off
        if wp.static("contact_force_hessian_vt" in VBD_DEBUG_PRINTING_OPTIONS):
            wp.printf(
                "v: %d dEdD: %f\nnormal force: %f %f %f\nfriction force: %f %f %f\n",
                v,
                dEdD,
                collision_force[0], collision_force[1], collision_force[2], friction_force[0], friction_force[1],
                friction_force[2],
            )
        # fmt: on

        displacement_0 = pos_anchor[tri_indices[tri, 0]] - a
        displacement_1 = pos_anchor[tri_indices[tri, 1]] - b
        displacement_2 = pos_anchor[tri_indices[tri, 2]] - c
        displacement_3 = pos_anchor[v] - p

        collision_force_0 = collision_force * bs[0]
        collision_force_1 = collision_force * bs[1]
        collision_force_2 = collision_force * bs[2]
        collision_force_3 = collision_force * bs[3]

        collision_hessian_0 = collision_hessian * bs[0] * bs[0]
        collision_hessian_1 = collision_hessian * bs[1] * bs[1]
        collision_hessian_2 = collision_hessian * bs[2] * bs[2]
        collision_hessian_3 = collision_hessian * bs[3] * bs[3]

        collision_normal_sign = wp.vec4(-1.0, -1.0, -1.0, 1.0)
        damping_force, damping_hessian = damp_collision(
            displacement_0,
            collision_normal * collision_normal_sign[0],
            collision_hessian_0,
            collision_damping,
            dt,
        )

        collision_force_0 += damping_force + bs[0] * friction_force
        collision_hessian_0 += damping_hessian + bs[0] * bs[0] * friction_hessian

        damping_force, damping_hessian = damp_collision(
            displacement_1,
            collision_normal * collision_normal_sign[1],
            collision_hessian_1,
            collision_damping,
            dt,
        )
        collision_force_1 += damping_force + bs[1] * friction_force
        collision_hessian_1 += damping_hessian + bs[1] * bs[1] * friction_hessian

        damping_force, damping_hessian = damp_collision(
            displacement_2,
            collision_normal * collision_normal_sign[2],
            collision_hessian_2,
            collision_damping,
            dt,
        )
        collision_force_2 += damping_force + bs[2] * friction_force
        collision_hessian_2 += damping_hessian + bs[2] * bs[2] * friction_hessian

        damping_force, damping_hessian = damp_collision(
            displacement_3,
            collision_normal * collision_normal_sign[3],
            collision_hessian_3,
            collision_damping,
            dt,
        )
        collision_force_3 += damping_force + bs[3] * friction_force
        collision_hessian_3 += damping_hessian + bs[3] * bs[3] * friction_hessian
        return (
            True,
            collision_force_0,
            collision_force_1,
            collision_force_2,
            collision_force_3,
            collision_hessian_0,
            collision_hessian_1,
            collision_hessian_2,
            collision_hessian_3,
        )
    else:
        collision_force = wp.vec3(0.0, 0.0, 0.0)
        collision_hessian = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        return (
            False,
            collision_force,
            collision_force,
            collision_force,
            collision_force,
            collision_hessian,
            collision_hessian,
            collision_hessian,
            collision_hessian,
        )



@wp.func
def evaluate_vertex_triangle_collision_force_hessian_4_vertices_log_collision(
    v: int,
    tri: int,
    pos: wp.array(dtype=wp.vec3),
    pos_anchor: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    collision_radius: float,
    collision_stiffness: float,
    collision_damping: float,
    friction_coefficient: float,
    friction_epsilon: float,
    dt: float,
):
    a = pos[tri_indices[tri, 0]]
    b = pos[tri_indices[tri, 1]]
    c = pos[tri_indices[tri, 2]]

    p = pos[v]

    closest_p, bary, _feature_type = triangle_closest_point(a, b, c, p)

    diff = p - closest_p
    dis = wp.length(diff)
    collision_normal = diff / dis

    if 0.0 < dis < collision_radius:
        bs = wp.vec4(-bary[0], -bary[1], -bary[2], 1.0)

        dEdD, d2E_dDdD = evaluate_self_contact_force_norm(dis, collision_radius, collision_stiffness)

        collision_force = -dEdD * collision_normal
        collision_hessian = d2E_dDdD * wp.outer(collision_normal, collision_normal)

        # friction force
        dx_v = p - pos_anchor[v] # pos_anchor is prev position

        closest_p_prev = (
            bary[0] * pos_anchor[tri_indices[tri, 0]]
            + bary[1] * pos_anchor[tri_indices[tri, 1]]
            + bary[2] * pos_anchor[tri_indices[tri, 2]]
        )

        dx = dx_v - (closest_p - closest_p_prev) # relative dx

        e0, e1 = build_orthonormal_basis(collision_normal)

        T = mat32(e0[0], e1[0], e0[1], e1[1], e0[2], e1[2])

        u = wp.transpose(T) * dx
        u_norm = wp.length(u)
        eps_U = friction_epsilon * dt

        friction_force, friction_hessian = compute_friction(friction_coefficient, -dEdD, T, u, eps_U)


        # fmt: off
        if wp.static("contact_force_hessian_vt" in VBD_DEBUG_PRINTING_OPTIONS):
            wp.printf(
                "v: %d dEdD: %f\nnormal force: %f %f %f\nfriction force: %f %f %f\n",
                v,
                dEdD,
                collision_force[0], collision_force[1], collision_force[2], friction_force[0], friction_force[1],
                friction_force[2],
            )
        # fmt: on

        displacement_0 = pos_anchor[tri_indices[tri, 0]] - a
        displacement_1 = pos_anchor[tri_indices[tri, 1]] - b
        displacement_2 = pos_anchor[tri_indices[tri, 2]] - c
        displacement_3 = pos_anchor[v] - p

        collision_force_0 = collision_force * bs[0] # tri_a
        collision_force_1 = collision_force * bs[1] # tri_b
        collision_force_2 = collision_force * bs[2] # tri_c
        collision_force_3 = collision_force * bs[3] # v

        contacts_normal_contact_force0 = collision_force_0
        contacts_normal_contact_force1 = collision_force_1
        contacts_normal_contact_force2 = collision_force_2
        contacts_normal_contact_force3 = collision_force_3

        collision_hessian_0 = collision_hessian * bs[0] * bs[0]
        collision_hessian_1 = collision_hessian * bs[1] * bs[1]
        collision_hessian_2 = collision_hessian * bs[2] * bs[2]
        collision_hessian_3 = collision_hessian * bs[3] * bs[3]

        collision_normal_sign = wp.vec4(-1.0, -1.0, -1.0, 1.0)
        damping_force, damping_hessian = damp_collision(
            displacement_0,
            collision_normal * collision_normal_sign[0],
            collision_hessian_0,
            collision_damping,
            dt,
        )
        ncfs = contacts_normal_contact_force0
        ncfm = contacts_normal_contact_force0

        collision_force_0 += damping_force + bs[0] * friction_force
        collision_hessian_0 += damping_hessian + bs[0] * bs[0] * friction_hessian

        contacts_friction0 = bs[0] * friction_force

        damping_force, damping_hessian = damp_collision(
            displacement_1,
            collision_normal * collision_normal_sign[1],
            collision_hessian_1,
            collision_damping,
            dt,
        )
        ncfs += contacts_normal_contact_force1
        if wp.length(contacts_normal_contact_force1) < wp.length(ncfm):
            ncfm = contacts_normal_contact_force1

        collision_force_1 += damping_force + bs[1] * friction_force
        collision_hessian_1 += damping_hessian + bs[1] * bs[1] * friction_hessian

        contacts_friction1 = bs[1] * friction_force

        damping_force, damping_hessian = damp_collision(
            displacement_2,
            collision_normal * collision_normal_sign[2],
            collision_hessian_2,
            collision_damping,
            dt,
        )
        ncfs += contacts_normal_contact_force2
        if wp.length(contacts_normal_contact_force2) < wp.length(ncfm):
            ncfm = contacts_normal_contact_force2
        collision_force_2 += damping_force + bs[2] * friction_force
        collision_hessian_2 += damping_hessian + bs[2] * bs[2] * friction_hessian

        contacts_friction2 = bs[2] * friction_force

        damping_force, damping_hessian = damp_collision(
            displacement_3,
            collision_normal * collision_normal_sign[3],
            collision_hessian_3,
            collision_damping,
            dt,
        )
        ncfs += contacts_normal_contact_force3
        if wp.length(contacts_normal_contact_force3) < wp.length(ncfm):
            ncfm = contacts_normal_contact_force3
        collision_force_3 += damping_force + bs[3] * friction_force
        collision_hessian_3 += damping_hessian + bs[3] * bs[3] * friction_hessian

        contacts_friction3 = bs[3] * friction_force

        return (
            True,
            collision_force_0,
            collision_force_1,
            collision_force_2,
            collision_force_3,
            collision_hessian_0,
            collision_hessian_1,
            collision_hessian_2,
            collision_hessian_3,
            T,
            collision_normal,
            u_norm,
            eps_U,
            friction_force,
            ncfs,
            ncfm,


            contacts_normal_contact_force0,
            contacts_normal_contact_force1,
            contacts_normal_contact_force2,
            contacts_normal_contact_force3,
            contacts_friction0,
            contacts_friction1,
            contacts_friction2,
            contacts_friction3
        )
    else:
        collision_force = wp.vec3(0.0, 0.0, 0.0)
        collision_hessian = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        return (
            False,
            collision_force,
            collision_force,
            collision_force,
            collision_force,
            collision_hessian,
            collision_hessian,
            collision_hessian,
            collision_hessian,
            mat32(0.0,0.0,0.0,0.0,0.0,0.0), # T
            collision_normal,
            0.0, # u_norm
            0.0, # eps_U
            collision_force, # friction_force
            collision_force, # ncfs
            collision_force, # ncfm

            collision_force,
            collision_force,
            collision_force,
            collision_force,
            collision_force,
            collision_force,
            collision_force,
            collision_force,
        )



@wp.func
def compute_friction(mu: float, normal_contact_force: float, T: mat32, u: wp.vec2, eps_u: float):
    """
    Returns the 1D friction force and hessian.
    Args:
        mu: Friction coefficient.
        normal_contact_force: normal contact force.
        T: Transformation matrix (3x2 matrix). 2D 접선좌표를 3D 월드로 올리는 기저(열벡터가 t,b)
        u: 2D displacement vector. 접선 평면에서의 누적 변위 혹은 이번 스텝에서의 접선 변위
    Outputs:
        force: 크기 3 벡터, 접선 마찰력(월드)
        hessian: 3*3 matrix, 선형화(뉴턴/implicit에 쓰기 위한)
    """
    # Friction
    u_norm = wp.length(u)

    if u_norm > 0.0:
        # IPC friction
        if u_norm > eps_u:
            # constant stage
            f1_SF_over_x = 1.0 / u_norm
        else:
            # smooth transition
            f1_SF_over_x = (-u_norm / eps_u + 2.0) / eps_u

        force = -mu * normal_contact_force * T * (f1_SF_over_x * u)

        # Different from IPC, we treat the contact normal as constant
        # this significantly improves the stability
        hessian = mu * normal_contact_force * T * (f1_SF_over_x * wp.identity(2, float)) * wp.transpose(T)
    else:
        force = wp.vec3(0.0, 0.0, 0.0)
        hessian = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    return force, hessian


@wp.kernel
def forward_step(
    dt: float,
    gravity: wp.array(dtype=wp.vec3),
    pos_prev: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    external_force: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    inertia: wp.array(dtype=wp.vec3),
):
    particle = wp.tid()
    pos_prev[particle] = pos[particle]

    if not particle_flags[particle] & ParticleFlags.ACTIVE:
        inertia[particle] = pos[particle]
        return
    vel_new = vel[particle] + (gravity[0] + external_force[particle] * inv_mass[particle]) * dt
    pos[particle] = pos[particle] + vel_new * dt
    inertia[particle] = pos[particle]


@wp.kernel
def forward_step_penetration_free(
    dt: float,
    gravity: wp.array(dtype=wp.vec3),
    pos_prev: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    external_force: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    pos_prev_collision_detection: wp.array(dtype=wp.vec3),
    particle_conservative_bounds: wp.array(dtype=float),
    inertia: wp.array(dtype=wp.vec3),
):
    """
    Forward integration step for particles (Penetration Free mode).
    """
    particle_index = wp.tid()
    pos_prev[particle_index] = pos[particle_index]

    if not particle_flags[particle_index] & ParticleFlags.ACTIVE:
        inertia[particle_index] = pos[particle_index]
        return

    vel_new = vel[particle_index] + (gravity[0] + external_force[particle_index] * inv_mass[particle_index]) * dt
    pos_inertia = pos[particle_index] + vel_new * dt
    inertia[particle_index] = pos_inertia

    pos[particle_index] = apply_conservative_bound_truncation(
        particle_index, pos_inertia, pos_prev_collision_detection, particle_conservative_bounds
    )


@wp.kernel
def compute_particle_conservative_bound(
    # inputs
    conservative_bound_relaxation: float,
    collision_query_radius: float,
    adjacency: ParticleForceElementAdjacencyInfo,
    collision_info: TriMeshCollisionInfo,
    # outputs
    particle_conservative_bounds: wp.array(dtype=float),
):
    particle_index = wp.tid()
    # min_dist = wp.min(collision_query_radius, collision_info.vertex_colliding_triangles_min_dist[particle_index])
    min_dist = collision_info.vertex_colliding_triangles_min_dist[particle_index]
    # bound from neighbor triangles
    for i_adj_tri in range(
        get_vertex_num_adjacent_faces(
            adjacency,
            particle_index,
        )
    ):
        tri_index, _vertex_order = get_vertex_adjacent_face_id_order(
            adjacency,
            particle_index,
            i_adj_tri,
        )
        min_dist = wp.min(min_dist, collision_info.triangle_colliding_vertices_min_dist[tri_index])

    # bound from neighbor edges
    for i_adj_edge in range(
        get_vertex_num_adjacent_edges(
            adjacency,
            particle_index,
        )
    ):
        nei_edge_index, vertex_order_on_edge = get_vertex_adjacent_edge_id_order(
            adjacency,
            particle_index,
            i_adj_edge,
        )
        # vertex is on the edge; otherwise it only effects the bending energy
        if vertex_order_on_edge == 2 or vertex_order_on_edge == 3:
            # collisions of neighbor edges
            min_dist = wp.min(min_dist, collision_info.edge_colliding_edges_min_dist[nei_edge_index])

    particle_conservative_bounds[particle_index] = conservative_bound_relaxation * min_dist


@wp.kernel
def validate_conservative_bound(
    pos: wp.array(dtype=wp.vec3),
    pos_prev_collision_detection: wp.array(dtype=wp.vec3),
    particle_conservative_bounds: wp.array(dtype=float),
):
    v_index = wp.tid()

    displacement = wp.length(pos[v_index] - pos_prev_collision_detection[v_index])

    if displacement > particle_conservative_bounds[v_index] * 1.01 and displacement > 1e-5:
        # wp.expect_eq(displacement <= particle_conservative_bounds[v_index] * 1.01, True)
        wp.printf(
            "Vertex %d has moved by %f exceeded the limit of %f\n",
            v_index,
            displacement,
            particle_conservative_bounds[v_index],
        )


@wp.func
def apply_conservative_bound_truncation(
    v_index: wp.int32,
    pos_new: wp.vec3,
    pos_prev_collision_detection: wp.array(dtype=wp.vec3),
    particle_conservative_bounds: wp.array(dtype=float),
):
    particle_pos_prev_collision_detection = pos_prev_collision_detection[v_index]
    accumulated_displacement = pos_new - particle_pos_prev_collision_detection
    conservative_bound = particle_conservative_bounds[v_index]

    accumulated_displacement_norm = wp.length(accumulated_displacement)
    if accumulated_displacement_norm > conservative_bound and conservative_bound > 1e-5:
        accumulated_displacement_norm_truncated = conservative_bound
        accumulated_displacement = accumulated_displacement * (
            accumulated_displacement_norm_truncated / accumulated_displacement_norm
        )

        return particle_pos_prev_collision_detection + accumulated_displacement
    else:
        return pos_new


@wp.kernel
def solve_trimesh_no_self_contact_tile(
    dt: float,
    particle_ids_in_color: wp.array(dtype=wp.int32),
    pos_anchor: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    inertia: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    tri_poses: wp.array(dtype=wp.mat22),
    tri_materials: wp.array(dtype=float, ndim=2),
    tri_areas: wp.array(dtype=float),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_rest_angles: wp.array(dtype=float),
    edge_rest_length: wp.array(dtype=float),
    edge_bending_properties: wp.array(dtype=float, ndim=2),
    adjacency: ParticleForceElementAdjacencyInfo,
    # contact info
    particle_forces: wp.array(dtype=wp.vec3),
    particle_hessians: wp.array(dtype=wp.mat33),
    # JGS2 cubature weights
    cubature_face_weights: wp.array(dtype=float),
    use_coord_condensation: int,
    # output
    pos_new: wp.array(dtype=wp.vec3),
    stvk_forces: wp.array(dtype=wp.vec3)
):
    tid = wp.tid()
    block_idx = tid // TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
    thread_idx = tid % TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
    particle_index = particle_ids_in_color[block_idx]

    if not particle_flags[particle_index] & ParticleFlags.ACTIVE:
        if thread_idx == 0:
            pos_new[particle_index] = pos[particle_index]
        return

    particle_pos = pos[particle_index]

    dt_sqr_reciprocal = 1.0 / (dt * dt)

    # # inertia force and hessian
    # f = mass[particle_index] * (inertia[particle_index] - pos[particle_index]) * (dt_sqr_reciprocal)
    # h = mass[particle_index] * dt_sqr_reciprocal * wp.identity(n=3, dtype=float)

    f = wp.vec3(0.0)
    h = wp.mat33(0.0)
    # JGS2 per-thread accumulators
    h_jgs2 = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    f_jgs2 = wp.vec3(0.0, 0.0, 0.0)

    num_adj_faces = get_vertex_num_adjacent_faces(adjacency, particle_index)
    adj_face_weight_base = adjacency.v_adj_faces_offsets[particle_index] >> 1

    batch_counter = wp.int32(0)

    # loop through all the adjacent triangles using whole block
    while batch_counter + thread_idx < num_adj_faces:
        adj_tri_counter = thread_idx + batch_counter
        batch_counter += TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
        # elastic force and hessian
        tri_index, vertex_order = get_vertex_adjacent_face_id_order(adjacency, particle_index, adj_tri_counter)

        f_tri, h_tri = evaluate_stvk_force_hessian(
            tri_index,
            vertex_order,
            pos,
            pos_anchor,
            tri_indices,
            tri_poses[tri_index],
            tri_areas[tri_index],
            tri_materials[tri_index, 0],
            tri_materials[tri_index, 1],
            tri_materials[tri_index, 2],
            dt,
        )
        stvk_forces[particle_index] += f_tri

        f += f_tri
        h += h_tri

        # JGS2 cubature correction
        w = cubature_face_weights[adj_face_weight_base + adj_tri_counter]
        h_jgs2 = h_jgs2 + w * h_tri
        f_jgs2 = f_jgs2 + w * f_tri

        # fmt: off
        if wp.static("elasticity_force_hessian" in VBD_DEBUG_PRINTING_OPTIONS):
            wp.printf(
                "particle: %d, i_adj_tri: %d, particle_order: %d, \nforce:\n %f %f %f, \nhessian:, \n%f %f %f, \n%f %f %f, \n%f %f %f\n",
                particle_index,
                thread_idx,
                vertex_order,
                f[0], f[1], f[2], h[0, 0], h[0, 1], h[0, 2], h[1, 0], h[1, 1], h[1, 2], h[2, 0], h[2, 1], h[2, 2],
            )
            # fmt: on

    batch_counter = wp.int32(0)
    num_adj_edges = get_vertex_num_adjacent_edges(adjacency, particle_index)
    while batch_counter + thread_idx < num_adj_edges:
        adj_edge_counter = batch_counter + thread_idx
        batch_counter += TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
        nei_edge_index, vertex_order_on_edge = get_vertex_adjacent_edge_id_order(
            adjacency, particle_index, adj_edge_counter
        )
        if edge_bending_properties[nei_edge_index, 0] != 0.0:
            f_edge, h_edge = evaluate_dihedral_angle_based_bending_force_hessian(
                nei_edge_index,
                vertex_order_on_edge,
                pos,
                pos_anchor,
                edge_indices,
                edge_rest_angles,
                edge_rest_length,
                edge_bending_properties[nei_edge_index, 0],
                edge_bending_properties[nei_edge_index, 1],
                dt,
            )
            stvk_forces[particle_index] += f_edge

            f += f_edge
            h += h_edge

    f_tile    = wp.tile(f,     preserve_type=True)
    h_tile    = wp.tile(h,     preserve_type=True)
    fj_tile   = wp.tile(f_jgs2, preserve_type=True)
    hj_tile   = wp.tile(h_jgs2, preserve_type=True)

    f_total   = wp.tile_reduce(wp.add, f_tile)[0]
    h_total   = wp.tile_reduce(wp.add, h_tile)[0]
    fj_total  = wp.tile_reduce(wp.add, fj_tile)[0]
    hj_total  = wp.tile_reduce(wp.add, hj_tile)[0]

    if thread_idx == 0:
        h_base = (
            h_total
            + mass[particle_index] * dt_sqr_reciprocal * wp.identity(n=3, dtype=float)
            + particle_hessians[particle_index]
        )
        f_final = (
            f_total
            + mass[particle_index] * (inertia[particle_index] - pos[particle_index]) * (dt_sqr_reciprocal)
            + particle_forces[particle_index]
            + fj_total   # gradient correction same for JGS2 and CoC
        )
        # JGS2 / CoC: branch on Hessian augmentation vs deflation
        h_final = h_base + hj_total  # default: JGS2
        if use_coord_condensation == 1:
            h_coc = h_base - hj_total
            if abs(wp.determinant(h_coc)) > 1e-5:
                h_final = h_coc
            # else: keep JGS2 fallback

        if abs(wp.determinant(h_final)) > 1e-5:
            h_inv = wp.inverse(h_final)
            pos_new[particle_index] = particle_pos + h_inv * f_final


@wp.kernel
def solve_trimesh_no_self_contact(
    dt: float,
    particle_ids_in_color: wp.array(dtype=wp.int32),
    pos_anchor: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    inertia: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    tri_poses: wp.array(dtype=wp.mat22),
    tri_materials: wp.array(dtype=float, ndim=2),
    tri_areas: wp.array(dtype=float),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_rest_angles: wp.array(dtype=float),
    edge_rest_length: wp.array(dtype=float),
    edge_bending_properties: wp.array(dtype=float, ndim=2),
    adjacency: ParticleForceElementAdjacencyInfo,
    # contact info
    particle_forces: wp.array(dtype=wp.vec3),
    particle_hessians: wp.array(dtype=wp.mat33),
    # JGS2 cubature weights (one float per vertex-adjacent-face entry)
    cubature_face_weights: wp.array(dtype=float),
    # 0 = JGS2 (augment Hessian), 1 = Coordinate Condensation (deflate Hessian)
    use_coord_condensation: int,
    # output
    pos_new: wp.array(dtype=wp.vec3),
    stvk_forces: wp.array(dtype=wp.vec3)
):
    tid = wp.tid()

    particle_index = particle_ids_in_color[tid]

    if not particle_flags[particle_index] & ParticleFlags.ACTIVE:
        pos_new[particle_index] = pos[particle_index]
        return

    particle_pos = pos[particle_index]

    dt_sqr_reciprocal = 1.0 / (dt * dt)

    # inertia force and hessian
    f = mass[particle_index] * (inertia[particle_index] - pos[particle_index]) * (dt_sqr_reciprocal)
    h = mass[particle_index] * dt_sqr_reciprocal * wp.identity(n=3, dtype=float)

    # JGS2 reduced Hessian / force accumulators (Lan et al. 2025, Eq. 15)
    h_jgs2 = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    f_jgs2 = wp.vec3(0.0, 0.0, 0.0)

    # elastic force and hessian
    adj_face_weight_base = adjacency.v_adj_faces_offsets[particle_index] >> 1
    for i_adj_tri in range(get_vertex_num_adjacent_faces(adjacency, particle_index)):
        tri_id, particle_order = get_vertex_adjacent_face_id_order(adjacency, particle_index, i_adj_tri)

        # fmt: off
        if wp.static("connectivity" in VBD_DEBUG_PRINTING_OPTIONS):
            wp.printf(
                "particle: %d | num_adj_faces: %d | ",
                particle_index,
                get_vertex_num_adjacent_faces(particle_index, adjacency),
            )
            wp.printf("i_face: %d | face id: %d | v_order: %d | ", i_adj_tri, tri_id, particle_order)
            wp.printf(
                "face: %d %d %d\n",
                tri_indices[tri_id, 0],
                tri_indices[tri_id, 1],
                tri_indices[tri_id, 2],
            )
        # fmt: on

        f_tri, h_tri = evaluate_stvk_force_hessian(
            tri_id,
            particle_order,
            pos,
            pos_anchor,
            tri_indices,
            tri_poses[tri_id],
            tri_areas[tri_id],
            tri_materials[tri_id, 0],
            tri_materials[tri_id, 1],
            tri_materials[tri_id, 2],
            dt,
        )
        stvk_forces[particle_index] += f_tri

        f = f + f_tri
        h = h + h_tri

        # JGS2: accumulate cubature-weighted correction
        w = cubature_face_weights[adj_face_weight_base + i_adj_tri]
        h_jgs2 = h_jgs2 + w * h_tri
        f_jgs2 = f_jgs2 + w * f_tri

        # fmt: off
        if wp.static("elasticity_force_hessian" in VBD_DEBUG_PRINTING_OPTIONS):
            wp.printf(
                "particle: %d, i_adj_tri: %d, particle_order: %d, \nforce:\n %f %f %f, \nhessian:, \n%f %f %f, \n%f %f %f, \n%f %f %f\n",
                particle_index,
                i_adj_tri,
                particle_order,
                f[0], f[1], f[2], h[0, 0], h[0, 1], h[0, 2], h[1, 0], h[1, 1], h[1, 2], h[2, 0], h[2, 1], h[2, 2],
            )
        # fmt: on

    for i_adj_edge in range(get_vertex_num_adjacent_edges(adjacency, particle_index)):
        nei_edge_index, vertex_order_on_edge = get_vertex_adjacent_edge_id_order(adjacency, particle_index, i_adj_edge)
        if edge_bending_properties[nei_edge_index, 0] != 0.0:
            f_edge, h_edge = evaluate_dihedral_angle_based_bending_force_hessian(
                nei_edge_index,
                vertex_order_on_edge,
                pos,
                pos_anchor,
                edge_indices,
                edge_rest_angles,
                edge_rest_length,
                edge_bending_properties[nei_edge_index, 0],
                edge_bending_properties[nei_edge_index, 1],
                dt,
            )
            stvk_forces[particle_index] += f_edge

            f += f_edge
            h += h_edge

    h += particle_hessians[particle_index]
    f += particle_forces[particle_index]

    # JGS2 / CoC: apply reduced Hessian H̃ and gradient correction f̃
    f_final = f + f_jgs2
    h_final = h + h_jgs2  # default: JGS2 (augment Hessian)
    if use_coord_condensation == 1:
        # CoC: deflate Hessian toward Newton Schur complement (H_ii - S, S ≈ H̃)
        h_coc = h - h_jgs2
        if abs(wp.determinant(h_coc)) > 1e-5:
            h_final = h_coc
        # else: h_final already set to JGS2 fallback above

    if abs(wp.determinant(h_final)) > 1e-5:
        hInv = wp.inverse(h_final)
        pos_new[particle_index] = particle_pos + hInv * f_final


@wp.kernel
def copy_particle_positions_back(
    particle_ids_in_color: wp.array(dtype=wp.int32),
    pos: wp.array(dtype=wp.vec3),
    pos_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    particle = particle_ids_in_color[tid]

    pos[particle] = pos_new[particle]


@wp.kernel
def update_velocity(
    dt: float, pos_prev: wp.array(dtype=wp.vec3), pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3)
):
    particle = wp.tid()
    vel[particle] = (pos[particle] - pos_prev[particle]) / dt


@wp.kernel
def convert_body_particle_contact_data_kernel(
    # inputs
    body_particle_contact_buffer_pre_alloc: int,
    soft_contact_particle: wp.array(dtype=int),
    contact_count: wp.array(dtype=int),
    contact_max: int,
    # outputs
    body_particle_contact_buffer: wp.array(dtype=int),
    body_particle_contact_count: wp.array(dtype=int),
):
    contact_index = wp.tid()
    count = min(contact_max, contact_count[0])
    if contact_index >= count:
        return

    particle_index = soft_contact_particle[contact_index]
    offset = particle_index * body_particle_contact_buffer_pre_alloc

    contact_counter = wp.atomic_add(body_particle_contact_count, particle_index, 1)
    if contact_counter < body_particle_contact_buffer_pre_alloc:
        body_particle_contact_buffer[offset + contact_counter] = contact_index


@wp.kernel
def accumulate_contact_force_and_hessian(
    # inputs
    dt: float,
    current_color: int,
    pos_anchor: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    particle_colors: wp.array(dtype=int),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    # self contact
    collision_info_array: wp.array(dtype=TriMeshCollisionInfo),
    collision_radius: float,
    soft_contact_ke: float,
    soft_contact_kd: float,
    friction_mu: float,
    friction_epsilon: float,
    edge_edge_parallel_epsilon: float,
    # body-particle contact
    particle_radius: wp.array(dtype=float),
    body_particle_contact_particle: wp.array(dtype=int),
    body_particle_contact_count: wp.array(dtype=int),
    body_particle_contact_max: int,
    # per-contact soft AVBD parameters for body-particle contacts (shared with rigid side)
    body_particle_contact_penalty_k: wp.array(dtype=float),
    body_particle_contact_material_kd: wp.array(dtype=float),
    body_particle_contact_material_mu: wp.array(dtype=float),
    shape_material_mu: wp.array(dtype=float),
    shape_body: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_q_prev: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    contact_shape: wp.array(dtype=int),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_body_vel: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    # outputs: particle force and hessian
    particle_forces: wp.array(dtype=wp.vec3),
    particle_hessians: wp.array(dtype=wp.mat33),
):
    t_id = wp.tid()
    collision_info = collision_info_array[0]

    primitive_id = t_id // NUM_THREADS_PER_COLLISION_PRIMITIVE
    t_id_current_primitive = t_id % NUM_THREADS_PER_COLLISION_PRIMITIVE

    # process edge-edge collisions
    if primitive_id < collision_info.edge_colliding_edges_buffer_sizes.shape[0]:
        e1_idx = primitive_id

        collision_buffer_counter = t_id_current_primitive
        collision_buffer_offset = collision_info.edge_colliding_edges_offsets[primitive_id]
        while collision_buffer_counter < collision_info.edge_colliding_edges_buffer_sizes[primitive_id]:
            e2_idx = collision_info.edge_colliding_edges[2 * (collision_buffer_offset + collision_buffer_counter) + 1]

            if e1_idx != -1 and e2_idx != -1:
                e1_v1 = edge_indices[e1_idx, 2]
                e1_v2 = edge_indices[e1_idx, 3]

                c_e1_v1 = particle_colors[e1_v1]
                c_e1_v2 = particle_colors[e1_v2]
                if c_e1_v1 == current_color or c_e1_v2 == current_color:
                    has_contact, collision_force_0, collision_force_1, collision_hessian_0, collision_hessian_1 = (
                        evaluate_edge_edge_contact_2_vertices(
                            e1_idx,
                            e2_idx,
                            pos,
                            pos_anchor,
                            edge_indices,
                            collision_radius,
                            soft_contact_ke,
                            soft_contact_kd,
                            friction_mu,
                            friction_epsilon,
                            dt,
                            edge_edge_parallel_epsilon,
                        )
                    )

                    if has_contact:
                        # here we only handle the e1 side, because e2 will also detection this contact and add force and hessian on its own
                        if c_e1_v1 == current_color:
                            wp.atomic_add(particle_forces, e1_v1, collision_force_0)
                            wp.atomic_add(particle_hessians, e1_v1, collision_hessian_0)
                        if c_e1_v2 == current_color:
                            wp.atomic_add(particle_forces, e1_v2, collision_force_1)
                            wp.atomic_add(particle_hessians, e1_v2, collision_hessian_1)
            collision_buffer_counter += NUM_THREADS_PER_COLLISION_PRIMITIVE

    # process vertex-triangle collisions
    if primitive_id < collision_info.vertex_colliding_triangles_buffer_sizes.shape[0]:
        particle_idx = primitive_id
        collision_buffer_counter = t_id_current_primitive
        collision_buffer_offset = collision_info.vertex_colliding_triangles_offsets[primitive_id]
        while collision_buffer_counter < collision_info.vertex_colliding_triangles_buffer_sizes[primitive_id]:
            tri_idx = collision_info.vertex_colliding_triangles[
                (collision_buffer_offset + collision_buffer_counter) * 2 + 1
            ]

            if particle_idx != -1 and tri_idx != -1:
                tri_a = tri_indices[tri_idx, 0]
                tri_b = tri_indices[tri_idx, 1]
                tri_c = tri_indices[tri_idx, 2]

                c_v = particle_colors[particle_idx]
                c_tri_a = particle_colors[tri_a]
                c_tri_b = particle_colors[tri_b]
                c_tri_c = particle_colors[tri_c]

                if (
                    c_v == current_color
                    or c_tri_a == current_color
                    or c_tri_b == current_color
                    or c_tri_c == current_color
                ):
                    (
                        has_contact,
                        collision_force_0,
                        collision_force_1,
                        collision_force_2,
                        collision_force_3,
                        collision_hessian_0,
                        collision_hessian_1,
                        collision_hessian_2,
                        collision_hessian_3,
                    ) = evaluate_vertex_triangle_collision_force_hessian_4_vertices(
                        particle_idx,
                        tri_idx,
                        pos,
                        pos_anchor,
                        tri_indices,
                        collision_radius,
                        soft_contact_ke,
                        soft_contact_kd,
                        friction_mu,
                        friction_epsilon,
                        dt,
                    )

                    if has_contact:
                        # particle
                        if c_v == current_color:
                            wp.atomic_add(particle_forces, particle_idx, collision_force_3)
                            wp.atomic_add(particle_hessians, particle_idx, collision_hessian_3)

                        # tri_a
                        if c_tri_a == current_color:
                            wp.atomic_add(particle_forces, tri_a, collision_force_0)
                            wp.atomic_add(particle_hessians, tri_a, collision_hessian_0)

                        # tri_b
                        if c_tri_b == current_color:
                            wp.atomic_add(particle_forces, tri_b, collision_force_1)
                            wp.atomic_add(particle_hessians, tri_b, collision_hessian_1)

                        # tri_c
                        if c_tri_c == current_color:
                            wp.atomic_add(particle_forces, tri_c, collision_force_2)
                            wp.atomic_add(particle_hessians, tri_c, collision_hessian_2)
            collision_buffer_counter += NUM_THREADS_PER_COLLISION_PRIMITIVE

    particle_body_contact_count = min(body_particle_contact_max, body_particle_contact_count[0])

    if t_id < particle_body_contact_count:
        particle_idx = body_particle_contact_particle[t_id]

        if particle_colors[particle_idx] == current_color:
            # Read per-contact AVBD penalty and material properties shared with the rigid side
            contact_ke = body_particle_contact_penalty_k[t_id]
            contact_kd = body_particle_contact_material_kd[t_id]
            contact_mu = body_particle_contact_material_mu[t_id]

            body_contact_force, body_contact_hessian = evaluate_body_particle_contact(
                particle_idx,
                pos[particle_idx],
                pos_anchor[particle_idx],
                t_id,
                contact_ke,
                contact_kd,
                contact_mu,
                friction_epsilon,
                particle_radius,
                shape_material_mu,
                shape_body,
                body_q,
                body_q_prev,
                body_qd,
                body_com,
                contact_shape,
                contact_body_pos,
                contact_body_vel,
                contact_normal,
                dt,
            )
            wp.atomic_add(particle_forces, particle_idx, body_contact_force)
            wp.atomic_add(particle_hessians, particle_idx, body_contact_hessian)


@wp.kernel
def accumulate_contact_force_and_hessian_log_collision(
    # inputs
    dt: float,
    current_color: int,
    pos_anchor: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    particle_colors: wp.array(dtype=int),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    # self contact
    collision_info_array: wp.array(dtype=TriMeshCollisionInfo),
    collision_radius: float,
    soft_contact_ke: float,
    soft_contact_kd: float,
    friction_mu: float,
    friction_epsilon: float,
    edge_edge_parallel_epsilon: float,
    # body-particle contact
    particle_radius: wp.array(dtype=float),
    body_particle_contact_particle: wp.array(dtype=int),
    body_particle_contact_count: wp.array(dtype=int),
    body_particle_contact_max: int,
    # per-contact soft AVBD parameters for body-particle contacts (shared with rigid side)
    body_particle_contact_penalty_k: wp.array(dtype=float),
    body_particle_contact_material_kd: wp.array(dtype=float),
    body_particle_contact_material_mu: wp.array(dtype=float),
    shape_material_mu: wp.array(dtype=float),
    shape_body: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_q_prev: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    contact_shape: wp.array(dtype=int),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_body_vel: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),

    collision_counter: wp.array(dtype=int),
    # outputs: particle force and hessian
    particle_forces: wp.array(dtype=wp.vec3),
    particle_hessians: wp.array(dtype=wp.mat33),

    contacts_index: wp.array(dtype=int),
    contacts_is_self_col: wp.array(dtype=bool),
    contacts_is_body_cloth_col: wp.array(dtype=bool),
    contacts_is_vt_col: wp.array(dtype=bool),
    contacts_is_ee_col: wp.array(dtype=bool),
    contacts_vid: wp.array(dtype=int),
    contacts_fid: wp.array(dtype=int),
    contacts_eid1: wp.array(dtype=int),
    contacts_eid2: wp.array(dtype=int),
    contacts_T: wp.array(dtype=wp.vec3),
    contacts_B: wp.array(dtype=wp.vec3),
    contacts_collision_normal: wp.array(dtype=wp.vec3),
    contacts_u_norm: wp.array(dtype=float),
    contacts_eps_u: wp.array(dtype=float),
    contacts_is_slip: wp.array(dtype=bool),
    contacts_friction_force: wp.array(dtype=wp.vec3),
    contacts_normal_contact_force_sum: wp.array(dtype=wp.vec3),
    contacts_normal_contact_force_min: wp.array(dtype=wp.vec3),
    
    contacts_normal_contact_force0: wp.array(dtype=wp.vec3),
    contacts_normal_contact_force1: wp.array(dtype=wp.vec3),
    contacts_normal_contact_force2: wp.array(dtype=wp.vec3),
    contacts_normal_contact_force3: wp.array(dtype=wp.vec3),
    contacts_friction0: wp.array(dtype=wp.vec3),
    contacts_friction1: wp.array(dtype=wp.vec3),
    contacts_friction2: wp.array(dtype=wp.vec3),
    contacts_friction3: wp.array(dtype=wp.vec3),
    contacts_v_list: wp.array(dtype=wp.vec4i),
    contacts_mu: wp.array(dtype=float)
):
    t_id = wp.tid()
    collision_info = collision_info_array[0]

    primitive_id = t_id // NUM_THREADS_PER_COLLISION_PRIMITIVE
    t_id_current_primitive = t_id % NUM_THREADS_PER_COLLISION_PRIMITIVE

    # process edge-edge collisions
    if primitive_id < collision_info.edge_colliding_edges_buffer_sizes.shape[0]:
        e1_idx = primitive_id

        collision_buffer_counter = t_id_current_primitive
        collision_buffer_offset = collision_info.edge_colliding_edges_offsets[primitive_id]
        while collision_buffer_counter < collision_info.edge_colliding_edges_buffer_sizes[primitive_id]:
            e2_idx = collision_info.edge_colliding_edges[2 * (collision_buffer_offset + collision_buffer_counter) + 1]

            if e1_idx != -1 and e2_idx != -1:
                e1_v1 = edge_indices[e1_idx, 2]
                e1_v2 = edge_indices[e1_idx, 3]

                c_e1_v1 = particle_colors[e1_v1]
                c_e1_v2 = particle_colors[e1_v2]
                if c_e1_v1 == current_color or c_e1_v2 == current_color:
                    (has_contact, collision_force_0, collision_force_1, collision_hessian_0, collision_hessian_1,
                    

                        T,
                        collision_normal,
                        u_norm,
                        eps_u,
                        friction_force,
                        normal_contact_force_sum,
                        normal_contact_force_min,

                        normal_contact_force0,
                        normal_contact_force1,
                        normal_contact_force2,
                        normal_contact_force3,
                        friction0,
                        friction1,
                        friction2,
                        friction3
                    ) = (
                        evaluate_edge_edge_contact_2_vertices_log_collision(
                            e1_idx,
                            e2_idx,
                            pos,
                            pos_anchor,
                            edge_indices,
                            collision_radius,
                            soft_contact_ke,
                            soft_contact_kd,
                            friction_mu,
                            friction_epsilon,
                            dt,
                            edge_edge_parallel_epsilon,
                        )
                    )

                    if has_contact:
                        # logging
                        index = wp.atomic_add(collision_counter, 0, 1)
                        if index < collision_counter[0]:
                            contacts_index[index] = index
                            contacts_is_self_col[index] = True
                            contacts_is_body_cloth_col[index] = False
                            contacts_is_vt_col[index] = False
                            contacts_is_ee_col[index] = True
                            contacts_vid[index] = -1
                            contacts_fid[index] = -1
                            contacts_eid1[index] = e1_v1
                            contacts_eid2[index] = e1_v2
                            contacts_T[index] = T[:,0]
                            contacts_B[index] = T[:,1]
                            contacts_collision_normal[index] = collision_normal
                            contacts_u_norm[index] = u_norm
                            contacts_eps_u[index] = eps_u
                            contacts_is_slip[index] = u_norm > eps_u
                            contacts_friction_force[index] = friction_force
                            contacts_normal_contact_force_sum[index] = normal_contact_force_sum
                            contacts_normal_contact_force_min[index] = normal_contact_force_min

                            contacts_normal_contact_force0[index] = normal_contact_force0
                            contacts_normal_contact_force1[index] = normal_contact_force1
                            contacts_normal_contact_force2[index] = normal_contact_force2
                            contacts_normal_contact_force3[index] = normal_contact_force3
                            contacts_friction0[index] = friction0
                            contacts_friction1[index] = friction1
                            contacts_friction2[index] = friction2
                            contacts_friction3[index] = friction3

                            contacts_v_list[index] = wp.vec4i(e1_v1, e1_v2, 0, 0)
                            contacts_mu[index] = friction_mu


                        # here we only handle the e1 side, because e2 will also detection this contact and add force and hessian on its own
                        if c_e1_v1 == current_color:
                            wp.atomic_add(particle_forces, e1_v1, collision_force_0)
                            wp.atomic_add(particle_hessians, e1_v1, collision_hessian_0)
                        if c_e1_v2 == current_color:
                            wp.atomic_add(particle_forces, e1_v2, collision_force_1)
                            wp.atomic_add(particle_hessians, e1_v2, collision_hessian_1)
            collision_buffer_counter += NUM_THREADS_PER_COLLISION_PRIMITIVE

    # process vertex-triangle collisions
    if primitive_id < collision_info.vertex_colliding_triangles_buffer_sizes.shape[0]:
        particle_idx = primitive_id
        collision_buffer_counter = t_id_current_primitive
        collision_buffer_offset = collision_info.vertex_colliding_triangles_offsets[primitive_id]
        while collision_buffer_counter < collision_info.vertex_colliding_triangles_buffer_sizes[primitive_id]:
            tri_idx = collision_info.vertex_colliding_triangles[
                (collision_buffer_offset + collision_buffer_counter) * 2 + 1
            ]

            if particle_idx != -1 and tri_idx != -1:
                tri_a = tri_indices[tri_idx, 0]
                tri_b = tri_indices[tri_idx, 1]
                tri_c = tri_indices[tri_idx, 2]

                c_v = particle_colors[particle_idx]
                c_tri_a = particle_colors[tri_a]
                c_tri_b = particle_colors[tri_b]
                c_tri_c = particle_colors[tri_c]

                if (
                    c_v == current_color
                    or c_tri_a == current_color
                    or c_tri_b == current_color
                    or c_tri_c == current_color
                ):
                    (
                        has_contact,
                        collision_force_0,
                        collision_force_1,
                        collision_force_2,
                        collision_force_3,
                        collision_hessian_0,
                        collision_hessian_1,
                        collision_hessian_2,
                        collision_hessian_3,

                        T,
                        collision_normal,
                        u_norm,
                        eps_u,
                        friction_force,
                        normal_contact_force_sum,
                        normal_contact_force_min,

                        normal_contact_force0,
                        normal_contact_force1,
                        normal_contact_force2,
                        normal_contact_force3,
                        friction0,
                        friction1,
                        friction2,
                        friction3
                    ) = evaluate_vertex_triangle_collision_force_hessian_4_vertices_log_collision(
                        particle_idx,
                        tri_idx,
                        pos,
                        pos_anchor,
                        tri_indices,
                        collision_radius,
                        soft_contact_ke,
                        soft_contact_kd,
                        friction_mu,
                        friction_epsilon,
                        dt,
                    )
                    
                    if has_contact:
                        # logging
                        index = wp.atomic_add(collision_counter, 0, 1)
                        if index < collision_counter[0]:
                            contacts_index[index] = index
                            contacts_is_self_col[index] = True
                            contacts_is_body_cloth_col[index] = False
                            contacts_is_vt_col[index] = True
                            contacts_is_ee_col[index] = False
                            contacts_vid[index] = particle_idx
                            contacts_fid[index] = tri_idx
                            contacts_eid1[index] = -1
                            contacts_eid2[index] = -1
                            contacts_T[index] = T[:,0]
                            contacts_B[index] = T[:,1]
                            contacts_collision_normal[index] = collision_normal
                            contacts_u_norm[index] = u_norm
                            contacts_eps_u[index] = eps_u
                            contacts_is_slip[index] = u_norm > eps_u
                            contacts_friction_force[index] = friction_force
                            contacts_normal_contact_force_sum[index] = normal_contact_force_sum
                            contacts_normal_contact_force_min[index] = normal_contact_force_min

                            contacts_normal_contact_force0[index] = normal_contact_force0
                            contacts_normal_contact_force1[index] = normal_contact_force1
                            contacts_normal_contact_force2[index] = normal_contact_force2
                            contacts_normal_contact_force3[index] = normal_contact_force3
                            contacts_friction0[index] = friction0
                            contacts_friction1[index] = friction1
                            contacts_friction2[index] = friction2
                            contacts_friction3[index] = friction3

                            contacts_v_list[index] = wp.vec4i(tri_a, tri_b, tri_c, particle_idx)
                            contacts_mu[index] = friction_mu
                        # particle
                        if c_v == current_color:
                            wp.atomic_add(particle_forces, particle_idx, collision_force_3)
                            wp.atomic_add(particle_hessians, particle_idx, collision_hessian_3)

                        # tri_a
                        if c_tri_a == current_color:
                            wp.atomic_add(particle_forces, tri_a, collision_force_0)
                            wp.atomic_add(particle_hessians, tri_a, collision_hessian_0)

                        # tri_b
                        if c_tri_b == current_color:
                            wp.atomic_add(particle_forces, tri_b, collision_force_1)
                            wp.atomic_add(particle_hessians, tri_b, collision_hessian_1)

                        # tri_c
                        if c_tri_c == current_color:
                            wp.atomic_add(particle_forces, tri_c, collision_force_2)
                            wp.atomic_add(particle_hessians, tri_c, collision_hessian_2)
            collision_buffer_counter += NUM_THREADS_PER_COLLISION_PRIMITIVE

    particle_body_contact_count = min(body_particle_contact_max, body_particle_contact_count[0])

    if t_id < particle_body_contact_count:
        particle_idx = body_particle_contact_particle[t_id]

        if particle_colors[particle_idx] == current_color:
            # Read per-contact AVBD penalty and material properties shared with the rigid side
            contact_ke = body_particle_contact_penalty_k[t_id]
            contact_kd = body_particle_contact_material_kd[t_id]
            contact_mu = body_particle_contact_material_mu[t_id]


            (body_contact_force, body_contact_hessian,
            Tv,
            Bv,
            collision_normal,
            u_norm,
            eps_u,
            friction_force,

            normal_contact_force0,
            normal_contact_force1,
            normal_contact_force2,
            normal_contact_force3,
            friction0,
            friction1,
            friction2,
            friction3,
            mu
            ) = evaluate_body_particle_contact_log_collision(
                particle_idx,
                pos[particle_idx],
                pos_anchor[particle_idx],
                t_id,
                contact_ke,
                contact_kd,
                contact_mu,
                friction_epsilon,
                particle_radius,
                shape_material_mu,
                shape_body,
                body_q,
                body_q_prev,
                body_qd,
                body_com,
                contact_shape,
                contact_body_pos,
                contact_body_vel,
                contact_normal,
                dt,
            )

            index = wp.atomic_add(collision_counter, 0, 1)
            if index < collision_counter[0]:
                contacts_index[index] = index
                contacts_is_self_col[index] = False
                contacts_is_body_cloth_col[index] = True
                contacts_is_vt_col[index] = False # both false means we don't know
                contacts_is_ee_col[index] = False
                contacts_vid[index] = particle_idx
                contacts_fid[index] = -1
                contacts_eid1[index] = -1
                contacts_eid2[index] = -1
                contacts_T[index] = Tv # both zero vec
                contacts_B[index] = Bv
                contacts_collision_normal[index] = collision_normal
                contacts_u_norm[index] = u_norm
                contacts_eps_u[index] = eps_u
                contacts_is_slip[index] = u_norm > eps_u
                contacts_friction_force[index] = friction_force
                contacts_normal_contact_force_sum[index] = normal_contact_force0
                contacts_normal_contact_force_min[index] = normal_contact_force0

                contacts_normal_contact_force0[index] = normal_contact_force0
                contacts_normal_contact_force1[index] = normal_contact_force1
                contacts_normal_contact_force2[index] = normal_contact_force2
                contacts_normal_contact_force3[index] = normal_contact_force3
                contacts_friction0[index] = friction0
                contacts_friction1[index] = friction1
                contacts_friction2[index] = friction2
                contacts_friction3[index] = friction3

                contacts_v_list[index] = wp.vec4i(particle_idx, 0, 0, 0)
                contacts_mu[index] = mu

            wp.atomic_add(particle_forces, particle_idx, body_contact_force)
            wp.atomic_add(particle_hessians, particle_idx, body_contact_hessian)




def _csr_row(vals: np.ndarray, offs: np.ndarray, i: int) -> np.ndarray:
    """Extract CSR row `i` from the flattened adjacency arrays."""
    return vals[offs[i] : offs[i + 1]]


def _set_to_csr(list_of_sets, dtype=np.int32, sort=True):
    """
    Convert a list of integer sets into CSR (Compressed Sparse Row) structure.
    Args:
        list_of_sets: Iterable where each entry is a set of ints.
        dtype: Output dtype for the flattened arrays.
        sort: Whether to sort each row when writing into `flat`.
    Returns:
        A tuple `(flat, offsets)` representing the CSR values and offsets.
    """
    offsets = np.zeros(len(list_of_sets) + 1, dtype=dtype)
    sizes = np.fromiter((len(s) for s in list_of_sets), count=len(list_of_sets), dtype=dtype)
    np.cumsum(sizes, out=offsets[1:])
    flat = np.empty(offsets[-1], dtype=dtype)
    idx = 0
    for s in list_of_sets:
        if sort:
            arr = np.fromiter(sorted(s), count=len(s), dtype=dtype)
        else:
            arr = np.fromiter(s, count=len(s), dtype=dtype)

        flat[idx : idx + len(arr)] = arr
        idx += len(arr)
    return flat, offsets


def one_ring_vertices(
    v: int, edge_indices: np.ndarray, v_adj_edges: np.ndarray, v_adj_edges_offsets: np.ndarray
) -> np.ndarray:
    """
    Find immediate neighboring vertices that share an edge with vertex `v`.
    Args:
        v: Vertex index whose neighborhood is queried.
        edge_indices: Array of shape [num_edges, 4] storing edge endpoint indices.
        v_adj_edges: Flattened CSR adjacency array listing edge ids and local order.
        v_adj_edges_offsets: CSR offsets indexing into `v_adj_edges`.
    Returns:
        Sorted array of neighboring vertex indices, excluding `v`.
    """
    e_u = edge_indices[:, 2]
    e_v = edge_indices[:, 3]
    # preserve only the adjacent edge information, remove the order information
    inc_edges = _csr_row(v_adj_edges, v_adj_edges_offsets, v)[::2]
    inc_edges_order = _csr_row(v_adj_edges, v_adj_edges_offsets, v)[1::2]
    if inc_edges.size == 0:
        return np.empty(0)
    us = e_u[inc_edges[np.where(inc_edges_order >= 2)]]
    vs = e_v[inc_edges[np.where(inc_edges_order >= 2)]]

    assert (np.logical_or(us == v, vs == v)).all()
    nbrs = np.unique(np.concatenate([us, vs]))
    return nbrs[nbrs != v]


def leq_n_ring_vertices(
    v: int, edge_indices: np.ndarray, n: int, v_adj_edges: np.ndarray, v_adj_edges_offsets: np.ndarray
) -> np.ndarray:
    """
    Find all vertices within n-ring distance of vertex v using BFS.
    Args:
        v: Starting vertex index
        edge_indices: Edge connectivity array
        n: Maximum ring distance
        v_adj_edges: CSR values for vertex-edge adjacency
        v_adj_edges_offsets: CSR offsets for vertex-edge adjacency
    Returns:
        Array of all vertices within n-ring distance, including v itself
    """
    visited = {v}
    frontier = {v}
    for _ in range(n):
        next_frontier = set()
        for u in frontier:
            for w in one_ring_vertices(u, edge_indices, v_adj_edges, v_adj_edges_offsets):  # iterable of neighbors of u
                if w not in visited:
                    visited.add(w)
                    next_frontier.add(w)
        if not next_frontier:
            break
        frontier = next_frontier
    return np.fromiter(visited, dtype=int)


def build_vertex_n_ring_tris_collision_filter(
    n: int,
    num_vertices: int,
    edge_indices: np.ndarray,
    v_adj_edges: np.ndarray,
    v_adj_edges_offsets: np.ndarray,
    v_adj_faces: np.ndarray,
    v_adj_faces_offsets: np.ndarray,
):
    """
    For each vertex v, return ONLY triangles adjacent to v's one ring neighbor vertices.
    Excludes triangles incident to v itself (dist 0).
    Returns:
      v_two_flat, v_two_offs: CSR of strict-2-ring triangle ids per vertex
    """

    if n <= 1:
        return None, None

    v_nei_tri_sets = [set() for _ in range(num_vertices)]

    for v in range(num_vertices):
        # distance-1 vertices

        if n == 2:
            ring_n_minus_1 = one_ring_vertices(v, edge_indices, v_adj_edges, v_adj_edges_offsets)
        else:
            ring_n_minus_1 = leq_n_ring_vertices(v, edge_indices, n - 1, v_adj_edges, v_adj_edges_offsets)

        ring_1_tri_set = set(_csr_row(v_adj_faces, v_adj_faces_offsets, v)[::2])

        nei_tri_set = v_nei_tri_sets[v]
        for w in ring_n_minus_1:
            if w != v:
                # preserve only the adjacent edge information, remove the order information
                nei_tri_set.update(_csr_row(v_adj_faces, v_adj_faces_offsets, w)[::2])

        nei_tri_set.difference_update(ring_1_tri_set)

    return v_nei_tri_sets


def build_edge_n_ring_edge_collision_filter(
    n: int,
    edge_indices: np.ndarray,
    v_adj_edges: np.ndarray,
    v_adj_edges_offsets: np.ndarray,
):
    """
    For each vertex v, return ONLY triangles adjacent to v's one ring neighbor vertices.
    Excludes triangles incident to v itself (dist 0).
    Returns:
      v_two_flat, v_two_offs: CSR of strict-2-ring triangle ids per vertex
    """

    if n <= 1:
        return None, None

    edge_nei_edge_sets = [set() for _ in range(edge_indices.shape[0])]

    for e_idx in range(edge_indices.shape[0]):
        # distance-1 vertices
        v1 = edge_indices[e_idx, 2]
        v2 = edge_indices[e_idx, 3]

        if n == 2:
            ring_n_minus_1_v1 = one_ring_vertices(v1, edge_indices, v_adj_edges, v_adj_edges_offsets)
            ring_n_minus_1_v2 = one_ring_vertices(v2, edge_indices, v_adj_edges, v_adj_edges_offsets)
        else:
            ring_n_minus_1_v1 = leq_n_ring_vertices(v1, edge_indices, n - 1, v_adj_edges, v_adj_edges_offsets)
            ring_n_minus_1_v2 = leq_n_ring_vertices(v2, edge_indices, n - 1, v_adj_edges, v_adj_edges_offsets)

        all_neighbors = set(ring_n_minus_1_v1)
        all_neighbors.update(ring_n_minus_1_v2)

        ring_1_edge_set = set(_csr_row(v_adj_edges, v_adj_edges_offsets, v1)[::2])
        ring_2_edge_set = set(_csr_row(v_adj_edges, v_adj_edges_offsets, v2)[::2])

        nei_edge_set = edge_nei_edge_sets[e_idx]
        for w in all_neighbors:
            if w != v1 and w != v2:
                # preserve only the adjacent edge information, remove the order information
                # nei_tri_set.update(_csr_row(v_adj_faces, v_adj_faces_offsets, w)[::2])
                adj_edges = _csr_row(v_adj_edges, v_adj_edges_offsets, w)[::2]
                adj_edges_order = _csr_row(v_adj_edges, v_adj_edges_offsets, w)[1::2]
                adj_collision_edges = adj_edges[np.where(adj_edges_order >= 2)]
                nei_edge_set.update(adj_collision_edges)

        nei_edge_set.difference_update(ring_1_edge_set)
        nei_edge_set.difference_update(ring_2_edge_set)

    return edge_nei_edge_sets


@wp.func
def evaluate_spring_force_and_hessian(
    particle_idx: int,
    spring_idx: int,
    dt: float,
    pos: wp.array(dtype=wp.vec3),
    pos_anchor: wp.array(dtype=wp.vec3),
    spring_indices: wp.array(dtype=int),
    spring_rest_length: wp.array(dtype=float),
    spring_stiffness: wp.array(dtype=float),
    spring_damping: wp.array(dtype=float),
):
    v0 = spring_indices[spring_idx * 2]
    v1 = spring_indices[spring_idx * 2 + 1]

    diff = pos[v0] - pos[v1]
    l = wp.length(diff)
    l0 = spring_rest_length[spring_idx]

    force_sign = 1.0 if particle_idx == v0 else -1.0

    spring_force = force_sign * spring_stiffness[spring_idx] * (l0 - l) / l * diff
    spring_hessian = spring_stiffness[spring_idx] * (
        wp.identity(3, float) - (l0 / l) * (wp.identity(3, float) - wp.outer(diff, diff) / (l * l))
    )

    # compute damping
    h_d = spring_hessian * (spring_damping[spring_idx] / dt)

    f_d = h_d * (pos_anchor[particle_idx] - pos[particle_idx])

    spring_force = spring_force + f_d
    spring_hessian = spring_hessian + h_d

    return spring_force, spring_hessian


@wp.kernel
def accumulate_spring_force_and_hessian(
    # inputs
    dt: float,
    pos_anchor: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    particle_ids_in_color: wp.array(dtype=int),
    adjacency: ParticleForceElementAdjacencyInfo,
    # spring constraints
    spring_indices: wp.array(dtype=int),
    spring_rest_length: wp.array(dtype=float),
    spring_stiffness: wp.array(dtype=float),
    spring_damping: wp.array(dtype=float),
    # outputs: particle force and hessian
    particle_forces: wp.array(dtype=wp.vec3),
    particle_hessians: wp.array(dtype=wp.mat33),
):
    t_id = wp.tid()

    particle_index = particle_ids_in_color[t_id]

    num_adj_springs = get_vertex_num_adjacent_springs(adjacency, particle_index)
    for spring_counter in range(num_adj_springs):
        spring_index = get_vertex_adjacent_spring_id(adjacency, particle_index, spring_counter)
        spring_force, spring_hessian = evaluate_spring_force_and_hessian(
            particle_index,
            spring_index,
            dt,
            pos,
            pos_anchor,
            spring_indices,
            spring_rest_length,
            spring_stiffness,
            spring_damping,
        )

        particle_forces[particle_index] = particle_forces[particle_index] + spring_force
        particle_hessians[particle_index] = particle_hessians[particle_index] + spring_hessian


@wp.kernel
def accumulate_contact_force_and_hessian_no_self_contact(
    # inputs
    dt: float,
    current_color: int,
    pos_anchor: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    particle_colors: wp.array(dtype=int),
    # body-particle contact
    friction_epsilon: float,
    particle_radius: wp.array(dtype=float),
    body_particle_contact_particle: wp.array(dtype=int),
    body_particle_contact_count: wp.array(dtype=int),
    body_particle_contact_max: int,
    # per-contact soft AVBD parameters for body-particle contacts (shared with rigid side)
    body_particle_contact_penalty_k: wp.array(dtype=float),
    body_particle_contact_material_kd: wp.array(dtype=float),
    body_particle_contact_material_mu: wp.array(dtype=float),
    shape_material_mu: wp.array(dtype=float),
    shape_body: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_q_prev: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    contact_shape: wp.array(dtype=int),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_body_vel: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    # outputs: particle force and hessian
    particle_forces: wp.array(dtype=wp.vec3),
    particle_hessians: wp.array(dtype=wp.mat33),
):
    t_id = wp.tid()

    particle_body_contact_count = min(body_particle_contact_max, body_particle_contact_count[0])

    if t_id < particle_body_contact_count:
        particle_idx = body_particle_contact_particle[t_id]

        if particle_colors[particle_idx] == current_color:
            # Read per-contact AVBD penalty and material properties shared with the rigid side
            contact_ke = body_particle_contact_penalty_k[t_id]
            contact_kd = body_particle_contact_material_kd[t_id]
            contact_mu = body_particle_contact_material_mu[t_id]

            body_contact_force, body_contact_hessian = evaluate_body_particle_contact(
                particle_idx,
                pos[particle_idx],
                pos_anchor[particle_idx],
                t_id,
                contact_ke,
                contact_kd,
                contact_mu,
                friction_epsilon,
                particle_radius,
                shape_material_mu,
                shape_body,
                body_q,
                body_q_prev,
                body_qd,
                body_com,
                contact_shape,
                contact_body_pos,
                contact_body_vel,
                contact_normal,
                dt,
            )
            wp.atomic_add(particle_forces, particle_idx, body_contact_force)
            wp.atomic_add(particle_hessians, particle_idx, body_contact_hessian)


@wp.kernel
def solve_trimesh_with_self_contact_penetration_free(
    dt: float,
    particle_ids_in_color: wp.array(dtype=wp.int32),
    pos_anchor: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    inertia: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    tri_poses: wp.array(dtype=wp.mat22),
    tri_materials: wp.array(dtype=float, ndim=2),
    tri_areas: wp.array(dtype=float),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_rest_angles: wp.array(dtype=float),
    edge_rest_length: wp.array(dtype=float),
    edge_bending_properties: wp.array(dtype=float, ndim=2),
    adjacency: ParticleForceElementAdjacencyInfo,
    particle_forces: wp.array(dtype=wp.vec3),
    particle_hessians: wp.array(dtype=wp.mat33),
    pos_prev_collision_detection: wp.array(dtype=wp.vec3),
    particle_conservative_bounds: wp.array(dtype=float),
    # output
    pos_new: wp.array(dtype=wp.vec3),
    stvk_forces: wp.array(dtype=wp.vec3)
):
    t_id = wp.tid()

    particle_index = particle_ids_in_color[t_id]
    particle_pos = pos[particle_index]

    if not particle_flags[particle_index] & ParticleFlags.ACTIVE:
        pos_new[particle_index] = particle_pos
        return

    dt_sqr_reciprocal = 1.0 / (dt * dt)

    # inertia force and hessian
    f = mass[particle_index] * (inertia[particle_index] - pos[particle_index]) * (dt_sqr_reciprocal)
    h = mass[particle_index] * dt_sqr_reciprocal * wp.identity(n=3, dtype=float)

    # fmt: off
    if wp.static("inertia_force_hessian" in VBD_DEBUG_PRINTING_OPTIONS):
        wp.printf(
            "particle: %d after accumulate inertia\nforce:\n %f %f %f, \nhessian:, \n%f %f %f, \n%f %f %f, \n%f %f %f\n",
            particle_index,
            f[0], f[1], f[2], h[0, 0], h[0, 1], h[0, 2], h[1, 0], h[1, 1], h[1, 2], h[2, 0], h[2, 1], h[2, 2],
        )

    # elastic force and hessian
    for i_adj_tri in range(get_vertex_num_adjacent_faces(adjacency, particle_index)):
        tri_index, vertex_order = get_vertex_adjacent_face_id_order(adjacency, particle_index, i_adj_tri)

        # fmt: off
        if wp.static("connectivity" in VBD_DEBUG_PRINTING_OPTIONS):
            wp.printf(
                "particle: %d | num_adj_faces: %d | ",
                particle_index,
                get_vertex_num_adjacent_faces(particle_index, adjacency),
            )
            wp.printf("i_face: %d | face id: %d | v_order: %d | ", i_adj_tri, tri_index, vertex_order)
            wp.printf(
                "face: %d %d %d\n",
                tri_indices[tri_index, 0],
                tri_indices[tri_index, 1],
                tri_indices[tri_index, 2],
            )
        # fmt: on

        f_tri, h_tri = evaluate_stvk_force_hessian(
            tri_index,
            vertex_order,
            pos,
            pos_anchor,
            tri_indices,
            tri_poses[tri_index],
            tri_areas[tri_index],
            tri_materials[tri_index, 0],
            tri_materials[tri_index, 1],
            tri_materials[tri_index, 2],
            dt,
        )
        stvk_forces[particle_index] += f_tri

        f = f + f_tri
        h = h + h_tri


    for i_adj_edge in range(get_vertex_num_adjacent_edges(adjacency, particle_index)):
        nei_edge_index, vertex_order_on_edge = get_vertex_adjacent_edge_id_order(adjacency, particle_index, i_adj_edge)
        # vertex is on the edge; otherwise it only effects the bending energy n
        if edge_bending_properties[nei_edge_index, 0] != 0.0:
            f_edge, h_edge = evaluate_dihedral_angle_based_bending_force_hessian(
                nei_edge_index, vertex_order_on_edge, pos, pos_anchor, edge_indices, edge_rest_angles, edge_rest_length,
                edge_bending_properties[nei_edge_index, 0],
                edge_bending_properties[nei_edge_index, 1],
                dt,
            )

            stvk_forces[particle_index] += f_edge

            f = f + f_edge
            h = h + h_edge

    # fmt: off
    if wp.static("overall_force_hessian" in VBD_DEBUG_PRINTING_OPTIONS):
        wp.printf(
            "vertex: %d final\noverall force:\n %f %f %f, \noverall hessian:, \n%f %f %f, \n%f %f %f, \n%f %f %f\n",
            particle_index,
            f[0], f[1], f[2], h[0, 0], h[0, 1], h[0, 2], h[1, 0], h[1, 1], h[1, 2], h[2, 0], h[2, 1], h[2, 2],
        )

    # # fmt: on
    h = h + particle_hessians[particle_index]
    f = f + particle_forces[particle_index]

    if abs(wp.determinant(h)) > 1e-5:
        h_inv = wp.inverse(h)
        particle_pos_new = pos[particle_index] + h_inv * f

        pos_new[particle_index] = apply_conservative_bound_truncation(
            particle_index, particle_pos_new, pos_prev_collision_detection, particle_conservative_bounds
        )


@wp.kernel
def solve_trimesh_with_self_contact_penetration_free_tile(
    dt: float,
    particle_ids_in_color: wp.array(dtype=wp.int32),
    pos_anchor: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    inertia: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    tri_poses: wp.array(dtype=wp.mat22),
    tri_materials: wp.array(dtype=float, ndim=2),
    tri_areas: wp.array(dtype=float),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_rest_angles: wp.array(dtype=float),
    edge_rest_length: wp.array(dtype=float),
    edge_bending_properties: wp.array(dtype=float, ndim=2),
    adjacency: ParticleForceElementAdjacencyInfo,
    particle_forces: wp.array(dtype=wp.vec3),
    particle_hessians: wp.array(dtype=wp.mat33),
    pos_prev_collision_detection: wp.array(dtype=wp.vec3),
    particle_conservative_bounds: wp.array(dtype=float),
    # output
    pos_new: wp.array(dtype=wp.vec3),
    stvk_forces: wp.array(dtype=wp.vec3)
):
    tid = wp.tid()
    block_idx = tid // TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
    thread_idx = tid % TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
    particle_index = particle_ids_in_color[block_idx]

    if not particle_flags[particle_index] & ParticleFlags.ACTIVE:
        if thread_idx == 0:
            pos_new[particle_index] = pos[particle_index]
        return

    particle_pos = pos[particle_index]

    dt_sqr_reciprocal = 1.0 / (dt * dt)

    # elastic force and hessian
    num_adj_faces = get_vertex_num_adjacent_faces(adjacency, particle_index)

    f = wp.vec3(0.0)
    h = wp.mat33(0.0)

    batch_counter = wp.int32(0)

    # loop through all the adjacent triangles using whole block
    while batch_counter + thread_idx < num_adj_faces:
        adj_tri_counter = thread_idx + batch_counter
        batch_counter += TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
        # elastic force and hessian
        tri_index, vertex_order = get_vertex_adjacent_face_id_order(adjacency, particle_index, adj_tri_counter)

        # fmt: off
        if wp.static("connectivity" in VBD_DEBUG_PRINTING_OPTIONS):
            wp.printf(
                "particle: %d | num_adj_faces: %d | ",
                particle_index,
                get_vertex_num_adjacent_faces(particle_index, adjacency),
            )
            wp.printf("i_face: %d | face id: %d | v_order: %d | ", adj_tri_counter, tri_index, vertex_order)
            wp.printf(
                "face: %d %d %d\n",
                tri_indices[tri_index, 0],
                tri_indices[tri_index, 1],
                tri_indices[tri_index, 2],
            )
        # fmt: on

        f_tri, h_tri = evaluate_stvk_force_hessian(
            tri_index,
            vertex_order,
            pos,
            pos_anchor,
            tri_indices,
            tri_poses[tri_index],
            tri_areas[tri_index],
            tri_materials[tri_index, 0],
            tri_materials[tri_index, 1],
            tri_materials[tri_index, 2],
            dt,
        )

        stvk_forces[particle_index] += f_tri

        f += f_tri
        h += h_tri

    batch_counter = wp.int32(0)
    num_adj_edges = get_vertex_num_adjacent_edges(adjacency, particle_index)
    while batch_counter + thread_idx < num_adj_edges:
        adj_edge_counter = batch_counter + thread_idx
        batch_counter += TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
        nei_edge_index, vertex_order_on_edge = get_vertex_adjacent_edge_id_order(
            adjacency, particle_index, adj_edge_counter
        )
        if edge_bending_properties[nei_edge_index, 0] != 0.0:
            f_edge, h_edge = evaluate_dihedral_angle_based_bending_force_hessian(
                nei_edge_index,
                vertex_order_on_edge,
                pos,
                pos_anchor,
                edge_indices,
                edge_rest_angles,
                edge_rest_length,
                edge_bending_properties[nei_edge_index, 0],
                edge_bending_properties[nei_edge_index, 1],
                dt,
            )
            stvk_forces[particle_index] += f_edge
            f += f_edge
            h += h_edge

    f_tile = wp.tile(f, preserve_type=True)
    h_tile = wp.tile(h, preserve_type=True)

    f_total = wp.tile_reduce(wp.add, f_tile)[0]
    h_total = wp.tile_reduce(wp.add, h_tile)[0]

    if thread_idx == 0:
        h_total = (
            h_total
            + mass[particle_index] * dt_sqr_reciprocal * wp.identity(n=3, dtype=float)
            + particle_hessians[particle_index]
        )
        if abs(wp.determinant(h_total)) > 1e-5:
            h_inv = wp.inverse(h_total)
            f_total = (
                f_total
                + mass[particle_index] * (inertia[particle_index] - pos[particle_index]) * (dt_sqr_reciprocal)
                + particle_forces[particle_index]
            )
            particle_pos_new = particle_pos + h_inv * f_total

            pos_new[particle_index] = apply_conservative_bound_truncation(
                particle_index, particle_pos_new, pos_prev_collision_detection, particle_conservative_bounds
            )

# ===== JGS2 Precomputation =====
def compute_schur_static_numpy(
    particle_count,
    pos_rest,       # (N, 3) numpy
    tri_indices,    # (T, 3) numpy
    tri_poses,      # (T, 2, 2) numpy
    tri_areas,      # (T,) numpy
    tri_materials,  # (T, 3) numpy  [mu, lambda, damping]
):
    """
    Rest-shape에서 static Schur complement 계산.
    S_i = K_ii - sum_{j in N(i)} K_ij @ inv(K_jj) @ K_ji
    """
    N = particle_count
    T = len(tri_indices)

    def get_hessian_blocks(tri_id):
        """삼각형 하나에서 3×3 vertex 쌍의 K_ij 블록 반환 (3,3,3,3)"""
        v0, v1, v2 = tri_indices[tri_id]
        x0 = pos_rest[v0]
        x01 = pos_rest[v1] - x0
        x02 = pos_rest[v2] - x0

        DmInv = tri_poses[tri_id]  # 2×2
        D00, D01 = DmInv[0, 0], DmInv[0, 1]
        D10, D11 = DmInv[1, 0], DmInv[1, 1]

        mu   = tri_materials[tri_id, 0]
        lmbd = tri_materials[tri_id, 1]
        area = tri_areas[tri_id]

        # Deformation gradient columns
        f0 = x01 * D00 + x02 * D10
        f1 = x01 * D01 + x02 * D11

        f00 = np.dot(f0, f0)
        f11 = np.dot(f1, f1)
        f01 = np.dot(f0, f1)

        I3 = np.eye(3)
        Ic = f00 + f11
        two_dpsi_dIc = -mu + (0.5 * Ic - 1.0) * lmbd

        # 삼각형 공통 항
        A00 = (lmbd * np.outer(f0, f0)
               + two_dpsi_dIc * I3
               + mu * (f00 * I3 + 2.0 * np.outer(f0, f0) + np.outer(f1, f1)))
        A11 = (lmbd * np.outer(f1, f1)
               + two_dpsi_dIc * I3
               + mu * (f11 * I3 + 2.0 * np.outer(f1, f1) + np.outer(f0, f0)))
        A01 = lmbd * np.outer(f0, f1) + mu * (f01 * I3 + np.outer(f1, f0))
        A01T = A01.T

        # 각 vertex의 df0, df1 스칼라
        # v_order 0: mask = (-1, -1), 1: (1, 0), 2: (0, 1)
        masks = [(-1, -1), (1, 0), (0, 1)]
        df0 = [D00 * m[0] + D10 * m[1] for m in masks]
        df1 = [D01 * m[0] + D11 * m[1] for m in masks]

        # K_ij for all (i,j) pairs
        H = np.zeros((3, 3, 3, 3))
        for i in range(3):
            for j in range(3):
                H[i, j] = (df0[i]*df0[j] * A00
                          + df1[i]*df1[j] * A11
                          + df0[i]*df1[j] * A01
                          + df1[i]*df0[j] * A01T) * area
        return H, (v0, v1, v2)

    # Step 1: K_ii (diagonal blocks) 계산
    K_diag = np.zeros((N, 3, 3))
    for tri_id in range(T):
        H, (v0, v1, v2) = get_hessian_blocks(tri_id)
        verts = [v0, v1, v2]
        for i_local in range(3):
            K_diag[verts[i_local]] += H[i_local, i_local]

    # DEBUG
    # get_hessian_blocks 검증
    # 단순 체크: H[i,i] diagonal이 양수인지
    for tri_id in range(min(5, T)):
        H, (v0, v1, v2) = get_hessian_blocks(tri_id)
        print(f"tri {tri_id}:")
        for i in range(3):
            print(f"  K_{i}{i} diag: {np.diag(H[i,i])}")
        for i in range(3):
            for j in range(3):
                if i != j:
                    print(f"  K_{i}{j} norm: {np.linalg.norm(H[i,j]):.4f}")
    # correction 크기 vs K_ii 크기 비교
    print(f"K_ii mean norm: {np.mean([np.linalg.norm(K_diag[i]) for i in range(100)]):.4f}")
                
    # Step 2: Schur complement 계산
    # S_i = K_ii - sum_j K_ij @ inv(K_jj) @ K_ji
    schur = np.zeros((N, 3, 3))
    for tri_id in range(T):
        H, (v0, v1, v2) = get_hessian_blocks(tri_id)
        verts = [v0, v1, v2]
        for i_local in range(3):
            i_g = verts[i_local]
            for j_local in range(3):
                if i_local == j_local:
                    continue
                j_g = verts[j_local]
                K_ij = H[i_local, j_local]  # 3×3
                K_jj = K_diag[j_g]          # 3×3
                if abs(np.linalg.det(K_jj)) > 1e-10:
                    K_jj_inv = np.linalg.inv(K_jj)
                    schur[i_g] -= K_ij @ K_jj_inv @ K_ij.T
    
    # clamping 전 값 출력 추가
    print(f"[JGS2] before clamping mean diag: {np.mean(schur[:, [0,1,2], [0,1,2]]):.4f}")
    print(f"[JGS2] before clamping min eigval: {min(np.linalg.eigvalsh(schur[i]).min() for i in range(min(N,100))):.4f}")
    
    # Step 3: PSD clamping (음수 고유값 제거)
    for i in range(N):
        eigvals, eigvecs = np.linalg.eigh(schur[i])
        eigvals = np.maximum(eigvals, 0.0)
        schur[i] = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    print(f"[JGS2] schur correction mean diag: {np.mean(schur[:, [0,1,2], [0,1,2]]):.4f}")
    print(f"[JGS2] any nan? {np.any(np.isnan(schur))}")

    return schur.astype(np.float32)


def _nnls_numpy(A, b, max_iter=2000, tol=1e-12):
    """Non-negative least squares via projected gradient descent.

    Solves  min_{w>=0} ||A w - b||^2  for small dense A (m x n, n ~ 6).
    """
    AtA = A.T @ A
    Atb = A.T @ b
    lip = np.linalg.norm(AtA, ord=2) + 1e-10   # Lipschitz constant
    step = 1.0 / lip
    w = np.maximum(np.linalg.lstsq(A, b, rcond=None)[0], 0.0)
    for _ in range(max_iter):
        grad = AtA @ w - Atb
        w_new = np.maximum(w - step * grad, 0.0)
        if np.linalg.norm(w_new - w) < tol:
            return w_new
        w = w_new
    return w


def compute_jgs2_precomputation_numpy(
    particle_count,
    pos_rest,            # (N, 3) numpy
    tri_indices,         # (T, 3) numpy
    tri_poses,           # (T, 2, 2) numpy
    tri_areas,           # (T,) numpy
    tri_materials,       # (T, 2+) numpy  [mu, lambda, ...]
    v_adj_faces,         # (M,) numpy int32 – flat (tri_id, v_order) pairs
    v_adj_faces_offsets, # (N+1,) numpy int32 – byte offsets into v_adj_faces
):
    """JGS2 pre-computation (Section 4-6 of Lan et al. 2025).

    Computes per-(vertex, adjacent-triangle) cubature weights w_e such that at
    runtime:

        H_tilde^k_i ≈ Σ_e  w_e · H^{e,k}_{i,i}
        f_tilde^k_i ≈ Σ_e  w_e · f^{e,k}_i

    The weights are trained on the rest-shape correction

        H̄_tilde_i = Σ_{j∈N(i)} K_ij · K_jj^{-1} · K_ij^T   (≥ 0, PSD)

    via non-negative least squares (NNLS) so that

        Σ_e  w_e · K^e_{i,i}  ≈  H̄_tilde_i

    Using current-configuration per-element Hessians at runtime implicitly
    captures co-rotation without an explicit polar-decomposition pass.

    Returns
    -------
    cubature_face_weights : float32 ndarray, shape (len(v_adj_faces)//2,)
        Flat array indexed as  v_adj_faces_offsets[i]//2 + j  for vertex i,
        j-th adjacent face.
    """
    N = particle_count
    T = len(tri_indices)

    # ------------------------------------------------------------------ #
    # Helper: rest-shape Hessian blocks for one triangle                   #
    # Returns H[3,3,3,3] where H[a,b] = K^e_{ab} (3×3 block)              #
    # ------------------------------------------------------------------ #
    def _hessian_blocks(tri_id):
        v0, v1, v2 = tri_indices[tri_id]
        x0 = pos_rest[v0]
        D = tri_poses[tri_id]        # 2×2
        D00, D01 = D[0, 0], D[0, 1]
        D10, D11 = D[1, 0], D[1, 1]
        mu   = tri_materials[tri_id, 0]
        lmbd = tri_materials[tri_id, 1]
        area = tri_areas[tri_id]

        f0 = (pos_rest[v1] - x0) * D00 + (pos_rest[v2] - x0) * D10
        f1 = (pos_rest[v1] - x0) * D01 + (pos_rest[v2] - x0) * D11

        f00 = np.dot(f0, f0)
        f11 = np.dot(f1, f1)
        f01 = np.dot(f0, f1)
        I3  = np.eye(3)
        Ic  = f00 + f11
        c   = -mu + (0.5 * Ic - 1.0) * lmbd

        A00 = lmbd * np.outer(f0, f0) + c * I3 + mu * (f00 * I3 + 2.0 * np.outer(f0, f0) + np.outer(f1, f1))
        A11 = lmbd * np.outer(f1, f1) + c * I3 + mu * (f11 * I3 + 2.0 * np.outer(f1, f1) + np.outer(f0, f0))
        A01 = lmbd * np.outer(f0, f1) + mu * (f01 * I3 + np.outer(f1, f0))

        df0 = np.array([D00*(-1)+D10*(-1), D00*1+D10*0, D00*0+D10*1])
        df1 = np.array([D01*(-1)+D11*(-1), D01*1+D11*0, D01*0+D11*1])

        H = np.zeros((3, 3, 3, 3))
        for a in range(3):
            for b in range(3):
                H[a, b] = (df0[a]*df0[b]*A00 + df1[a]*df1[b]*A11
                           + df0[a]*df1[b]*A01 + df1[a]*df0[b]*A01.T) * area
        return H, (v0, v1, v2)

    # ------------------------------------------------------------------ #
    # Step 1: K_diag[i] = Σ_{e∋i} K^e_{ii}  (3×3 per vertex)             #
    # ------------------------------------------------------------------ #
    K_diag = np.zeros((N, 3, 3))
    # Also store per-triangle diagonal blocks for later NNLS use
    tri_diag = {}   # tri_id -> list[(v_global, local_idx, K^e_{ii})]
    for tri_id in range(T):
        H, verts = _hessian_blocks(tri_id)
        tri_diag[tri_id] = []
        for loc, vg in enumerate(verts):
            K_diag[vg] += H[loc, loc]
            tri_diag[tri_id].append((vg, loc, H[loc, loc].copy()))

    # ------------------------------------------------------------------ #
    # Step 2: H̄_tilde[i] = Σ_{j∈N(i)} K_ij · K_jj^{-1} · K_ij^T (≥ 0) #
    # ------------------------------------------------------------------ #
    H_tilde_bar = np.zeros((N, 3, 3))
    for tri_id in range(T):
        H, (v0, v1, v2) = _hessian_blocks(tri_id)
        verts = [v0, v1, v2]
        for a in range(3):
            i_g = verts[a]
            for b in range(3):
                if a == b:
                    continue
                j_g = verts[b]
                K_ij = H[a, b]
                K_jj = K_diag[j_g]
                # Use pseudo-inverse: handles singular Hessian (e.g. z-direction in flat cloth)
                K_jj_pinv = np.linalg.pinv(K_jj, rcond=1e-6)
                H_tilde_bar[i_g] += K_ij @ K_jj_pinv @ K_ij.T   # + sign (JGS2 Eq. 15)

    # ------------------------------------------------------------------ #
    # Step 3: NNLS cubature – find w_e ≥ 0 per (vertex, adj-tri) such    #
    # that  Σ_e w_e · K^e_{i,i} ≈ H̄_tilde[i]                           #
    # ------------------------------------------------------------------ #
    total_adj_face_entries = len(v_adj_faces) // 2  # one float per (vertex, adj-face) pair
    cubature_weights = np.zeros(total_adj_face_entries, dtype=np.float32)

    for i in range(N):
        off   = int(v_adj_faces_offsets[i])
        off1  = int(v_adj_faces_offsets[i + 1])
        n_adj = (off1 - off) >> 1
        if n_adj == 0:
            continue

        target = H_tilde_bar[i]         # 3×3
        target_norm = np.linalg.norm(target)
        if target_norm < 1e-12:
            continue

        # Build design matrix A ∈ R^{9 × n_adj}
        A_cols = []
        for j in range(n_adj):
            tri_id  = int(v_adj_faces[off + 2 * j])
            v_order = int(v_adj_faces[off + 2 * j + 1])
            H, _ = _hessian_blocks(tri_id)
            A_cols.append(H[v_order, v_order].flatten())  # K^e_{i,i} flattened

        A = np.column_stack(A_cols)   # (9, n_adj)
        b = target.flatten()          # (9,)

        w = _nnls_numpy(A, b)

        weight_off = off >> 1         # v_adj_faces_offsets[i] // 2
        cubature_weights[weight_off : weight_off + n_adj] = w.astype(np.float32)

    nnz = np.count_nonzero(cubature_weights)
    print(f"[JGS2] cubature weights: nnz={nnz}/{total_adj_face_entries}, "
          f"max={cubature_weights.max():.4f}, mean_pos={cubature_weights[cubature_weights>0].mean() if nnz else 0:.4f}")
    return cubature_weights
