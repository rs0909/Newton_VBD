


from __future__ import annotations

import numpy as np
import warp as wp
from warp.types import float32, matrix

from newton._src.solvers.vbd.rigid_vbd_kernels import evaluate_body_particle_contact, evaluate_body_particle_contact_log_collision

from ...geometry import ParticleFlags
from ...geometry.kernels import triangle_closest_point
from ..vbd.tri_mesh_collision import TriMeshCollisionInfo

from ..vbd.particle_vbd_kernels import evaluate_stvk_force_hessian,get_vertex_num_adjacent_edges,get_vertex_num_adjacent_faces,get_vertex_adjacent_face_id_order,VBD_DEBUG_PRINTING_OPTIONS,get_vertex_adjacent_edge_id_order,evaluate_dihedral_angle_based_bending_force_hessian


NUM_THREADS_PER_COLLISION_PRIMITIVE = 4
TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE = 16


@wp.kernel
def trimesh_with_self_contact(
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
    # h = h + particle_hessians[particle_index]
    # f = f + particle_forces[particle_index]
    
    particle_hessians[particle_index] += h
    particle_forces[particle_index] += f

    # if abs(wp.determinant(h)) > 1e-5:
    #     h_inv = wp.inverse(h)
    #     particle_pos_new = pos[particle_index] + h_inv * f

    #     pos_new[particle_index] = apply_conservative_bound_truncation(
    #         particle_index, particle_pos_new, pos_prev_collision_detection, particle_conservative_bounds
    #     )


@wp.kernel
def project_Hessian_to_SPD():
    pass