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
RES_U = 128
RES_V = 128

class mat32(matrix(shape=(3, 2), dtype=float32)):
    pass


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


# --- simple 64-bit mix hash (splitmix64 style) ---
@wp.func
def hash_u64(x: wp.uint64) -> wp.uint64:
    # SplitMix64-like mix (constants are standard)
    x = x + wp.uint64(0x9E3779B97F4A7C15)
    z = x
    z = (z ^ (z >> wp.uint64(30))) * wp.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> wp.uint64(27))) * wp.uint64(0x94D049BB133111EB)
    z = z ^ (z >> wp.uint64(31))
    return z

@wp.func
def clamp_int(x: int, lo: int, hi: int) -> int:
    return wp.max(lo, wp.min(hi, x))

@wp.func
def make_nonzero_key(key: wp.uint64) -> wp.uint64:
    # Ensure key != 0 because 0 means empty slot
    return key | wp.uint64(1)

# NOTE: Warp atomic CAS API naming may differ:
# - Some versions: wp.atomic_cas(arr, idx, expected, desired) -> old_value
# - Some versions: wp.atomic_compare_exchange(arr, idx, expected, desired) -> old_value
# Replace wp.atomic_cas(...) below with your Warp atomic CAS call.
@wp.func
def hashmap_find_or_insert_pair_id(
    key_in: wp.uint64,
    hash_keys: wp.array(dtype=wp.uint64),
    hash_vals: wp.array(dtype=wp.int32),
    pair_count: wp.array(dtype=wp.int32),
    pair_q: wp.array(dtype=wp.vec2),
    pair_t: wp.array(dtype=wp.vec3),
    pair_age: wp.array(dtype=wp.int32),
    H: int,
    Pmax: int,
    init_t: wp.vec3,         # initial tangent axis (any vector not parallel to normal)
    init_age: int,
    max_probe: int,
) -> int:
    key = make_nonzero_key(key_in)

    h = hash_u64(key)
    start = int(h % wp.uint64(H))

    for i in range(max_probe):
        slot = start + i
        if slot >= H:
            slot -= H

        k = hash_keys[slot]

        # Empty slot -> try to claim
        if k == wp.uint64(0):
            # Attempt to atomically set key
            # old = wp.atomic_cas(hash_keys, slot, 0, key)
            old = wp.atomic_cas(hash_keys, slot, wp.uint64(0), key)  # <-- adjust API if needed
            if old == wp.uint64(0):
                # We won the insertion: allocate new pair_id
                # pid = wp.atomic_add(pair_count, 0, 1)
                pid = wp.atomic_add(pair_count, 0, 1)  # <-- adjust API if needed
                if pid < Pmax:
                    hash_vals[slot] = pid

                    # initialize pair state
                    pair_q[pid] = wp.vec2(0.0, 0.0)
                    pair_t[pid] = init_t
                    pair_age[pid] = init_age
                    return pid
                else:
                    # Out of pair storage; mark slot invalid or just return -1
                    # (Could also reset hash_keys[slot]=0, but racey; better to pre-size Pmax)
                    return -1

            # If CAS failed, someone else inserted concurrently; fall through and re-check

        # Occupied slot: check key match
        if k == key:
            return hash_vals[slot]

    # Hash table too full / probe limit reached
    return -1

@wp.func
def uv_to_cell_id(uv: wp.vec2, res_u: int, res_v: int) -> int:
    # clamp uv into [0, 1-eps] to avoid hitting res boundary
    u = wp.max(0.0, wp.min(0.999999, uv[0]))
    v = wp.max(0.0, wp.min(0.999999, uv[1]))

    ix = int(u * float(res_u))
    iy = int(v * float(res_v))

    ix = clamp_int(ix, 0, res_u - 1)
    iy = clamp_int(iy, 0, res_v - 1)
    return ix + res_u * iy

@wp.func
def lerp_uv(a: wp.vec2, b: wp.vec2, t: float) -> wp.vec2:
    return a + (b - a) * t

@wp.func
def ee_cells_from_edges(
    e1: int,
    e2: int,
    s: float,
    t: float,
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    uvs: wp.array(dtype=wp.vec2),
    res_u: int,
    res_v: int,
) -> wp.vec2i:
    # edge_indices[e,2], [e,3] are endpoints in your code
    e1_v1 = edge_indices[e1, 2]
    e1_v2 = edge_indices[e1, 3]
    e2_v1 = edge_indices[e2, 2]
    e2_v2 = edge_indices[e2, 3]

    uv1 = lerp_uv(uvs[e1_v1], uvs[e1_v2], s)
    uv2 = lerp_uv(uvs[e2_v1], uvs[e2_v2], t)

    c1 = uv_to_cell_id(uv1, res_u, res_v)
    c2 = uv_to_cell_id(uv2, res_u, res_v)

    return wp.vec2i(c1, c2)

@wp.func
def canonicalize_pair(layerA: int, cellA: int, layerB: int, cellB: int):
    # Ensure (A) <= (B) in lexicographic order to merge duplicates
    if (layerA > layerB) or (layerA == layerB and cellA > cellB):
        tmpL = layerA
        tmpC = cellA
        layerA = layerB
        cellA = cellB
        layerB = tmpL
        cellB = tmpC
    return layerA, cellA, layerB, cellB

@wp.func
def pack_pair_key(
    type_id: int,   # 0:EE, 1:VF, 2:Body, etc.
    layerA: int,
    cellA: int,
    layerB: int,
    cellB: int,
) -> wp.uint64:
    # bit layout (example):
    # [ type:4 | layerA:8 | cellA:16 | layerB:8 | cellB:16 | unused:12 ]
    # cells need 14 bits for 128*128=16384, we store 16 for simplicity
    la = wp.uint64(layerA & 0xFF)
    lb = wp.uint64(layerB & 0xFF)
    ca = wp.uint64(cellA & 0xFFFF)
    cb = wp.uint64(cellB & 0xFFFF)
    ty = wp.uint64(type_id & 0xF)

    key = (ty << wp.uint64(60)) | (la << wp.uint64(52)) | (ca << wp.uint64(36)) | (lb << wp.uint64(28)) | (cb << wp.uint64(12))
    return make_nonzero_key(key)

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
    # NEW
    pid: int,
    pair_t: wp.array(dtype=wp.vec3),
    transport_eps: float,
    iter_num: int,
    max_iter_num: int,
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
        # axis_1, axis_2 = build_orthonormal_basis(collision_normal)
        t_old = wp.vec3(1,0,0)
        if pid >= 0:
            t_old = pair_t[pid]
        axis_1, axis_2 = tangent_transport(t_old, collision_normal, transport_eps)

        # ✅ pair_t 업데이트(최소한 마지막 iter에서만)
        if pid >= 0 and iter_num == max_iter_num:
            pair_t[pid] = axis_1

        T = mat32(
            axis_1[0],
            axis_2[0],
            axis_1[1],
            axis_2[1],
            axis_1[2],
            axis_2[2],
        )
        # u_eff = q_mem + u_sub
        u = wp.transpose(T) * dx
        eps_U = friction_epsilon * dt

        # fmt: off
        if wp.static("contact_force_hessian_ee" in VBD_DEBUG_PRINTING_OPTIONS):
            wp.printf(
                "    collision force:\n    %f %f %f,\n    collision hessian:\n    %f %f %f,\n    %f %f %f,\n    %f %f %f\n",
                collision_force[0], collision_force[1], collision_force[2], collision_hessian[0, 0], collision_hessian[0, 1], collision_hessian[0, 2], collision_hessian[1, 0], collision_hessian[1, 1], collision_hessian[1, 2], collision_hessian[2, 0], collision_hessian[2, 1], collision_hessian[2, 2],
            )
        # fmt: on

        # ✅ transport만 테스트: u 그대로 사용
        friction_force, friction_hessian = compute_friction(friction_coefficient, -dEdD, T, u, eps_U)

        # ✅ pair_t 업데이트(최소한 마지막 iter에서만)
        if pid >= 0 and iter_num == max_iter_num:
            pair_t[pid] = axis_1

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
    # NEW
    pid: int,
    pair_t: wp.array(dtype=wp.vec3),
    transport_eps: float,
    iter_num: int,
    max_iter_num: int,
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

        # axis_1, axis_2 = build_orthonormal_basis(collision_normal)
        t_old = wp.vec3(1,0,0)
        if pid >= 0:
            t_old = pair_t[pid]
        axis_1, axis_2 = tangent_transport(t_old, collision_normal, transport_eps)
        # ✅ pair_t 업데이트(최소한 마지막 iter에서만)
        if pid >= 0 and iter_num == max_iter_num:
            pair_t[pid] = axis_1

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

        # ✅ transport만 테스트: u 그대로 사용
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
    # NEW
    pid: int,
    pair_t: wp.array(dtype=wp.vec3),
    transport_eps: float,
    iter_num: int,
    max_iter_num: int,
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
        # OLD:
        # axis_1, axis_2 = build_orthonormal_basis(collision_normal)
        # NEW: transport
        t_old = wp.vec3(1,0,0)
        if pid >= 0:
            t_old = pair_t[pid]
        axis_1, axis_2 = tangent_transport(t_old, collision_normal, transport_eps)

        # ✅ pair_t 업데이트(최소한 마지막 iter에서만)
        if pid >= 0 and iter_num == max_iter_num:
            pair_t[pid] = axis_1

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

        # ✅ transport만 테스트: u 그대로 사용
        friction_force, friction_hessian = compute_friction(friction_coefficient, -dEdD, T, u, eps_U)

        # ✅ pair_t 업데이트(최소한 마지막 iter에서만)
        if pid >= 0 and iter_num == max_iter_num:
            pair_t[pid] = axis_1
            
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
                wp.vec3(0.0, 0.0, 0.0),
        )
    else:
        # no contact
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
    # NEW
    pid: int,
    pair_t: wp.array(dtype=wp.vec3),
    transport_eps: float,
    iter_num: int,
    max_iter_num: int,
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
        
        # e0, e1 = build_orthonormal_basis(collision_normal)
        # tangent transport
        t_old = wp.vec3(1.0, 0.0, 0.0)
        if pid >= 0:
            t_old = pair_t[pid]

        e0, e1 = tangent_transport(t_old, collision_normal, transport_eps)
        # 커밋(마지막 iter에서만)
        if pid >= 0 and iter_num == max_iter_num:
            pair_t[pid] = e0

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
    # NEW
    pid: int,
    pair_t: wp.array(dtype=wp.vec3),
    transport_eps: float,
    iter_num: int,
    max_iter_num: int,
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

        # e0, e1 = build_orthonormal_basis(collision_normal)
        # tangent transport
        t_old = wp.vec3(1.0, 0.0, 0.0)
        if pid >= 0:
            t_old = pair_t[pid]

        e0, e1 = tangent_transport(t_old, collision_normal, transport_eps)
        # 커밋(마지막 iter에서만)
        if pid >= 0 and iter_num == max_iter_num:
            pair_t[pid] = e0

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

    num_adj_faces = get_vertex_num_adjacent_faces(adjacency, particle_index)

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
        # compute damping
        stvk_forces[particle_index] += f_tri

        f += f_tri
        h += h_tri

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

    #
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

            pos_new[particle_index] = particle_pos + h_inv * f_total


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

    # elastic force and hessian
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

    if abs(wp.determinant(h)) > 1e-5:
        hInv = wp.inverse(h)
        pos_new[particle_index] = particle_pos + hInv * f


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
    
    # NEW
    ee_entry_pair_id: wp.array(dtype=wp.int32),
    vf_entry_pair_id: wp.array(dtype=wp.int32),
    transport_eps: float,
    ee_pair_t: wp.array(dtype=wp.vec3),
    vf_pair_t: wp.array(dtype=wp.vec3),
    iter_num: int,
    max_iter_num: int,

    # outputs: particle force and hessian
    particle_forces: wp.array(dtype=wp.vec3),
    particle_hessians: wp.array(dtype=wp.mat33),
):
    t_id = wp.tid()
    collision_info = collision_info_array[0]

    primitive_id = t_id // NUM_THREADS_PER_COLLISION_PRIMITIVE
    t_id_current_primitive = t_id % NUM_THREADS_PER_COLLISION_PRIMITIVE
    entry_id = collision_buffer_offset + collision_buffer_counter
    ee_pid = -1
    if entry_id < ee_entry_pair_id.shape[0]:
        ee_pid = ee_entry_pair_id[entry_id]

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
                            
                            #  New
                            ee_pid,
                            ee_pair_t,
                            transport_eps,
                            iter_num,
                            max_iter_num,
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
            entry_id = collision_buffer_offset + collision_buffer_counter
            vf_pid = -1
            if entry_id < vf_entry_pair_id.shape[0]:
                vf_pid = vf_entry_pair_id[entry_id]

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
                        vf_pid, vf_pair_t, transport_eps, iter_num, max_iter_num
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

    # NEW
    ee_entry_pair_id: wp.array(dtype=wp.int32),
    vf_entry_pair_id: wp.array(dtype=wp.int32),

    transport_eps: float,
    # # NEW: UVs + grid
    # uvs: wp.array(dtype=wp.vec2),
    # res_u: int,
    # res_v: int,

    # # NEW: layer ids (self-contact이면 동일)
    # layerA: int,
    # layerB: int,

    # # NEW: pair hash table + pair state arrays
    # hash_keys: wp.array(dtype=wp.uint64),
    # hash_vals: wp.array(dtype=wp.int32),
    # pair_count: wp.array(dtype=wp.int32),
    # pair_q: wp.array(dtype=wp.vec2),
    ee_pair_t: wp.array(dtype=wp.vec3),
    vf_pair_t: wp.array(dtype=wp.vec3),
    # pair_age: wp.array(dtype=wp.int32),
    # hash_capacity: int,
    # pair_capacity: int,
    # max_probe: int,

    # NEW: iteration info (q commit timing)
    iter_num: int,
    max_iter_num: int,

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

    entry_id = collision_buffer_offset + collision_buffer_counter
    ee_pid = -1
    if entry_id < ee_entry_pair_id.shape[0]:
        ee_pid = ee_entry_pair_id[entry_id]

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
                        friction3,
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
                            #  New
                            ee_pid,
                            ee_pair_t,
                            transport_eps,
                            iter_num,
                            max_iter_num,
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
            entry_id = collision_buffer_offset + collision_buffer_counter
            vf_pid = -1
            if entry_id < vf_entry_pair_id.shape[0]:
                vf_pid = vf_entry_pair_id[entry_id]

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
                        vf_pid, vf_pair_t, transport_eps, iter_num, max_iter_num
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
    
    # # NEW
    # ee_entry_pair_id: wp.array(dtype=wp.int32),
    # transport_eps: float,
    # # NEW: UVs + grid
    # uvs: wp.array(dtype=wp.vec2),
    # res_u: int,
    # res_v: int,

    # # NEW: layer ids (self-contact이면 동일)
    # layerA: int,
    # layerB: int,

    # # NEW: pair hash table + pair state arrays
    # hash_keys: wp.array(dtype=wp.uint64),
    # hash_vals: wp.array(dtype=wp.int32),
    # pair_count: wp.array(dtype=wp.int32),
    # pair_q: wp.array(dtype=wp.vec2),
    # pair_t: wp.array(dtype=wp.vec3),
    # pair_age: wp.array(dtype=wp.int32),
    # hash_capacity: int,
    # pair_capacity: int,
    # max_probe: int,

    # # NEW: iteration info (q commit timing)
    # iter_num: int,
    # max_iter_num: int,

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

#######################
'''
[assign_pair_ids]
방식: 이 커널은 collision detection 직후에, contact list 길이만큼 병렬로 돌면 돼.
입력: 
- trimesh_collision_info, uv, edge_indices, tri_indices 등
출력:
- contact_pair_id[contact_idx]
- pair table 갱신: key insert/lookup, age=CONTACT, (optional) last_seen_iter 업데이트


[decay_pairs]
방식: substep 끝(또는 iter 0 끝)에서
pair_age를 줄이고(dead 처리) / last_seen가 갱신 안 된 pair는 age-- / age==0이면 q reset


[uv_to_cell]

'''

@wp.func
def safe_normalize(v: wp.vec3, eps: float) -> wp.vec3:
    n = wp.length(v)
    if n > eps:
        return v / n
    return wp.vec3(1.0, 0.0, 0.0)

@wp.func
def tangent_transport(t_old: wp.vec3, n_new: wp.vec3, eps: float):
    # project old tangent onto new normal plane
    t_proj = t_old - wp.dot(t_old, n_new) * n_new
    t_new = safe_normalize(t_proj, eps)

    # if projection degenerate (t_old almost parallel to n_new), fall back
    # by constructing any orthonormal basis from n_new
    if wp.length(t_proj) <= eps:
        # your existing helper
        t_new, b_tmp = build_orthonormal_basis(n_new)
        b_new = b_tmp
    else:
        b_new = wp.cross(n_new, t_new)
        b_new = safe_normalize(b_new, eps)

    return t_new, b_new

@wp.kernel
@wp.kernel
def assign_pair_ids_vf_from_vt_list(
    # collision info
    collision_info_array: wp.array(dtype=TriMeshCollisionInfo),

    # geometry
    pos: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    uvs: wp.array(dtype=wp.vec2),

    # params
    layerA: int,
    layerB: int,
    res_u: int,
    res_v: int,

    # hash table + pair storage (VF도 EE랑 같은 pair 테이블을 써도 되고, 따로 써도 됨)
    hash_keys: wp.array(dtype=wp.uint64),
    hash_vals: wp.array(dtype=wp.int32),
    pair_count: wp.array(dtype=wp.int32),
    pair_t: wp.array(dtype=wp.vec3),
    pair_age: wp.array(dtype=wp.int32),
    hash_capacity: int,
    pair_capacity: int,
    max_probe: int,
    init_age: int,

    # output: per entry_id pair id
    vf_entry_pair_id: wp.array(dtype=wp.int32),
):
    t_id = wp.tid()
    collision_info = collision_info_array[0]

    primitive_id = t_id // NUM_THREADS_PER_COLLISION_PRIMITIVE
    t_id_current_primitive = t_id % NUM_THREADS_PER_COLLISION_PRIMITIVE

    # primitive_id corresponds to particle_idx (vertex)
    if primitive_id >= collision_info.vertex_colliding_triangles_buffer_sizes.shape[0]:
        return

    v = primitive_id

    collision_buffer_counter = t_id_current_primitive
    collision_buffer_offset = collision_info.vertex_colliding_triangles_offsets[primitive_id]
    buffer_size = collision_info.vertex_colliding_triangles_buffer_sizes[primitive_id]

    while collision_buffer_counter < buffer_size:
        entry_id = collision_buffer_offset + collision_buffer_counter

        # bounds check for storage
        if entry_id >= vf_entry_pair_id.shape[0]:
            collision_buffer_counter += NUM_THREADS_PER_COLLISION_PRIMITIVE
            continue

        tri = collision_info.vertex_colliding_triangles[entry_id * 2 + 1]

        if v != -1 and tri != -1:
            # vertex-side cell
            uv_v = uvs[v]
            cell_v = uv_to_cell_id(uv_v, res_u, res_v)

            # triangle vertices
            i0 = tri_indices[tri, 0]
            i1 = tri_indices[tri, 1]
            i2 = tri_indices[tri, 2]

            a = pos[i0]
            b = pos[i1]
            c = pos[i2]
            p = pos[v]

            # closest point barycentric
            closest_p, bary, _ft = triangle_closest_point(a, b, c, p)

            # face-side UV at closest point
            uv_f = uvs[i0] * bary[0] + uvs[i1] * bary[1] + uvs[i2] * bary[2]
            cell_f = uv_to_cell_id(uv_f, res_u, res_v)

            la, ca, lb, cb = canonicalize_pair(layerA, cell_v, layerB, cell_f)

            # type_id = 1 for VF
            key = pack_pair_key(1, la, ca, lb, cb)

            init_t = wp.vec3(1.0, 0.0, 0.0)

            pid = hashmap_find_or_insert_pair_id(
                key,
                hash_keys, hash_vals, pair_count,
                pair_t, pair_age,
                hash_capacity, pair_capacity,
                init_t, init_age,
                max_probe
            )

            vf_entry_pair_id[entry_id] = pid
            if pid >= 0:
                pair_age[pid] = init_age
        else:
            vf_entry_pair_id[entry_id] = -1

        collision_buffer_counter += NUM_THREADS_PER_COLLISION_PRIMITIVE

# NUM_THREADS_PER_COLLISION_PRIMITIVE 는 너 코드에 이미 정의돼 있다고 가정

@wp.kernel
def assign_pair_ids_ee_from_collision_info(
    # collision info
    collision_info_array: wp.array(dtype=TriMeshCollisionInfo),

    # geometry
    pos: wp.array(dtype=wp.vec3),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    uvs: wp.array(dtype=wp.vec2),

    # params
    layerA: int,
    layerB: int,
    res_u: int,
    res_v: int,
    edge_edge_parallel_epsilon: float,

    # hash table + pair storage
    hash_keys: wp.array(dtype=wp.uint64),
    hash_vals: wp.array(dtype=wp.int32),
    pair_count: wp.array(dtype=wp.int32),
    pair_t: wp.array(dtype=wp.vec3),
    pair_age: wp.array(dtype=wp.int32),
    hash_capacity: int,
    pair_capacity: int,
    max_probe: int,
    init_age: int,

    # output: per "entry" pair id (entry index = collision_buffer_offset + collision_buffer_counter)
    ee_entry_pair_id: wp.array(dtype=wp.int32),
):
    t_id = wp.tid()
    collision_info = collision_info_array[0]

    primitive_id = t_id // NUM_THREADS_PER_COLLISION_PRIMITIVE
    t_id_current_primitive = t_id % NUM_THREADS_PER_COLLISION_PRIMITIVE

    # Each primitive_id corresponds to an "e1_idx" in your accumulate kernel
    if primitive_id >= collision_info.edge_colliding_edges_buffer_sizes.shape[0]:
        return

    e1_idx = primitive_id

    collision_buffer_counter = t_id_current_primitive
    collision_buffer_offset = collision_info.edge_colliding_edges_offsets[primitive_id]
    buffer_size = collision_info.edge_colliding_edges_buffer_sizes[primitive_id]

    while collision_buffer_counter < buffer_size:
        entry_id = collision_buffer_offset + collision_buffer_counter
        if entry_id >= ee_entry_pair_id.shape[0]:
            collision_buffer_counter += NUM_THREADS_PER_COLLISION_PRIMITIVE
            continue
        # edge_colliding_edges stores pairs (e1, e2) but you only read the 2nd int in your code:
        # [2*entry + 0] = e1 (often redundant), [2*entry + 1] = e2
        e2_idx = collision_info.edge_colliding_edges[2 * entry_id + 1]

        if e1_idx != -1 and e2_idx != -1:
            # endpoints
            e1_v1 = edge_indices[e1_idx, 2]
            e1_v2 = edge_indices[e1_idx, 3]
            e2_v1 = edge_indices[e2_idx, 2]
            e2_v2 = edge_indices[e2_idx, 3]

            p1 = pos[e1_v1]
            q1 = pos[e1_v2]
            p2 = pos[e2_v1]
            q2 = pos[e2_v2]

            # closest params s,t (recomputed, same as evaluate)
            st = wp.closest_point_edge_edge(p1, q1, p2, q2, edge_edge_parallel_epsilon)
            s = st[0]
            t = st[1]

            # uv along edges
            uv1 = lerp_uv(uvs[e1_v1], uvs[e1_v2], s)
            uv2 = lerp_uv(uvs[e2_v1], uvs[e2_v2], t)

            cell1 = uv_to_cell_id(uv1, res_u, res_v)
            cell2 = uv_to_cell_id(uv2, res_u, res_v)

            la, ca, lb, cb = canonicalize_pair(layerA, cell1, layerB, cell2)

            # type_id=0 for EE
            key = pack_pair_key(0, la, ca, lb, cb)

            init_t = wp.vec3(1.0, 0.0, 0.0)

            pid = hashmap_find_or_insert_pair_id(
                key,
                hash_keys, hash_vals, pair_count,
                pair_t, pair_age,
                hash_capacity, pair_capacity,
                init_t, init_age,
                max_probe
            )

            ee_entry_pair_id[entry_id] = pid
            if pid >= 0:
                pair_age[pid] = init_age
        else:
            ee_entry_pair_id[entry_id] = -1

        collision_buffer_counter += NUM_THREADS_PER_COLLISION_PRIMITIVE

@wp.kernel
def decay_pairs(
        
    ):
    

    return
