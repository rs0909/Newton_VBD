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

"""Dynamic same-color proximity detection and watchlist barrier for OGC stability.

Addresses two structural vulnerabilities in the OGC isotropic conservative bound:

1. Simultaneous multi-vertex movement
   OGC guarantees penetration-free motion for a *single* vertex moving in isolation,
   but NOT for same-color-group vertices moving simultaneously.  Two vertices vi, vj
   in the same color can each move by their full OGC bound toward the other, producing
   a combined approach of r_vi + r_vj that exceeds the safe gap.

2. Staleness
   OGC bounds are computed once at substep start and reused across all Gauss-Seidel
   iterations.  As vertices move during iterations the actual minimum distances shrink,
   but the bounds remain calibrated to the initial configuration.

Mitigation zones (per same-color pair (vi, vj))
------------------------------------------------
    r_vi = particle_conservative_bounds[vi]   (OGC scalar bound per vertex)
    r_vj = particle_conservative_bounds[vj]

    R_recolor(i,j)   = r_vi + r_vj            -- combined OGC bounds
    R_watchlist(i,j) = 2 * R_recolor(i,j)     -- outer warning zone

Recolor zone   d(vi,vj) ≤ R_recolor:
    Both vertices are flagged for dynamic recoloring.  They are moved to separate
    greedy-colored dynamic groups and solved *sequentially* (one after the other),
    eliminating the simultaneous-movement risk.  No watchlist barrier needed here
    because sequential solve already prevents double-push.

Watchlist zone R_recolor < d(vi,vj) ≤ R_watchlist:
    The pair is cached in each vertex's watchlist at substep start.
    Every Gauss-Seidel iteration recomputes an IPC log-barrier from the *current*
    positions (state_in.particle_q, updated each color via copy_particle_positions_back),
    fixing the staleness issue.
    Mass-weighted half-stiffness  alpha_i = m_j/(m_i+m_j)  prevents double-push
    oscillation: each vertex only contributes its momentum-fraction of the repulsion
    so the combined effect equals one full barrier without energy injection.

Usage (once per substep, in _update_proximity_state)
----------------------------------------------------
    1. compute_vertex_aabbs_for_recolor   -- set vertex BVH bounds
    2. reset_proximity_state              -- zero watchlist counts + recolor flags
    3. find_same_color_proximity          -- populate watchlist and recolor flags
    Per Gauss-Seidel iteration:
    4. accumulate_watchlist_barrier_forces -- inject barrier into particle_forces/hessians
"""

from __future__ import annotations

import warp as wp

# Maximum watchlist partners stored per vertex.  Pairs beyond this limit are
# silently dropped; the OGC bound still provides the hard safety guarantee.
WATCHLIST_MAX_PER_VERTEX = wp.constant(wp.int32(16))


@wp.kernel
def compute_vertex_aabbs_for_recolor(
    pos: wp.array(dtype=wp.vec3),
    particle_conservative_bounds: wp.array(dtype=float),
    lower: wp.array(dtype=wp.vec3),
    upper: wp.array(dtype=wp.vec3),
):
    """Compute per-vertex AABBs for the watchlist BVH (dtype=vec3, matching wp.Bvh API).

    AABB for vertex v: [pos_v - 2*r_v,  pos_v + 2*r_v]  where r_v = r_OGC[v].

    The watchlist condition is d(vi,vj) <= 2*(r_vi+r_vj).  Building each leaf's AABB
    with half-radius 2*r_v means an overlap query from vi with the same AABB finds all
    vj where |pos_vi - pos_vj|_inf <= 2*r_vi + 2*r_vj -- a superset of the L2-ball.
    False positives are filtered by the exact L2 check inside find_same_color_proximity.

    Args:
        pos: Vertex positions at substep start (pos_prev_collision_detection).
        particle_conservative_bounds: OGC scalar bound r per vertex.
        lower: Output AABB lower corners, shape (N,), dtype vec3.
        upper: Output AABB upper corners, shape (N,), dtype vec3.
    """
    v = wp.tid()
    r = particle_conservative_bounds[v] * float(2.0)
    p = pos[v]
    lower[v] = p - wp.vec3(r, r, r)
    upper[v] = p + wp.vec3(r, r, r)


@wp.kernel
def reset_proximity_state(
    watchlist_count: wp.array(dtype=wp.int32),
    recolor_flags: wp.array(dtype=wp.int32),
):
    """Zero watchlist partner counts and recolor flags before proximity detection.

    Args:
        watchlist_count: Per-vertex count of registered watchlist partners.
        recolor_flags:   Per-vertex flag; > 0 means vertex needs dynamic recoloring.
    """
    v = wp.tid()
    watchlist_count[v] = wp.int32(0)
    recolor_flags[v] = wp.int32(0)


@wp.kernel
def find_same_color_proximity(
    pos: wp.array(dtype=wp.vec3),
    particle_colors: wp.array(dtype=wp.int32),
    particle_conservative_bounds: wp.array(dtype=float),
    bvh_id: wp.uint64,
    # outputs
    recolor_flags: wp.array(dtype=wp.int32),
    watchlist_partners: wp.array(dtype=wp.int32),
    watchlist_count: wp.array(dtype=wp.int32),
):
    """Detect same-color proximity and populate the watchlist + recolor flags.

    Each vertex vi queries the BVH with its watchlist AABB and classifies every
    same-color vertex vj it finds:

    Recolor zone   (d <= r_vi + r_vj):
        Both vi and vj are flagged for dynamic recoloring (atomic on vj).

    Watchlist zone (r_vi+r_vj < d <= 2*(r_vi+r_vj)):
        vj is appended to vi's watchlist.  vj's own thread independently finds vi
        and adds it to vj's watchlist -- no cross-write needed.

    Recolor-zone pairs are also registered in the watchlist so the barrier applies
    in the iterations before the recoloring takes full sequential effect.

    Args:
        pos: Vertex positions at substep start.
        particle_colors: Original graph-coloring color per vertex.
        particle_conservative_bounds: OGC scalar bound r per vertex.
        bvh_id: Handle for the vertex BVH built from compute_vertex_aabbs_for_recolor.
        recolor_flags: Output per-vertex flag (>0 = needs recoloring).
        watchlist_partners: Flat [vi * WATCHLIST_MAX + k] = k-th partner of vi.
        watchlist_count: Per-vertex count of registered partners (capped at WATCHLIST_MAX).
    """
    vi = wp.tid()
    pos_vi = pos[vi]
    r_vi = particle_conservative_bounds[vi]
    color_vi = particle_colors[vi]

    r_q = r_vi * float(2.0)
    query = wp.bvh_query_aabb(
        bvh_id,
        pos_vi - wp.vec3(r_q, r_q, r_q),
        pos_vi + wp.vec3(r_q, r_q, r_q),
    )

    vj = int(0)
    while wp.bvh_query_next(query, vj):
        if vj == vi:
            continue
        if particle_colors[vj] != color_vi:
            continue

        r_vj = particle_conservative_bounds[vj]
        r_recolor = r_vi + r_vj
        r_watchlist = float(2.0) * r_recolor

        dist = wp.length(pos[vj] - pos_vi)
        if dist > r_watchlist:
            continue

        if dist <= r_recolor:
            # Recolor zone: mark both for sequential solve
            recolor_flags[vi] = wp.int32(1)
            wp.atomic_add(recolor_flags, vj, wp.int32(1))

        # Watchlist zone (superset of recolor zone): cache for per-iteration barrier
        k = wp.atomic_add(watchlist_count, vi, wp.int32(1))
        if k < WATCHLIST_MAX_PER_VERTEX:
            watchlist_partners[vi * WATCHLIST_MAX_PER_VERTEX + k] = vj


@wp.kernel
def accumulate_watchlist_barrier_forces(
    particle_ids_in_color: wp.array(dtype=wp.int32),
    pos: wp.array(dtype=wp.vec3),
    particle_mass: wp.array(dtype=float),
    watchlist_partners: wp.array(dtype=wp.int32),
    watchlist_count: wp.array(dtype=wp.int32),
    particle_conservative_bounds: wp.array(dtype=float),
    barrier_stiffness: float,
    # outputs
    particle_forces: wp.array(dtype=wp.vec3),
    particle_hessians: wp.array(dtype=wp.mat33),
):
    """Accumulate IPC log-barrier forces for watchlist pairs onto the current color.

    Barrier energy (activated only in the watchlist zone, not recolor zone):
        B(d) = -k * d_hat^2 * ln(d / d_hat)    for  0 < d < d_hat
        B(d) = 0                                 for  d >= d_hat
        d_hat = R_watchlist(vi, vj) = 2*(r_vi + r_vj)

    Negative gradient (repulsive force on vi):
        f_i = alpha_i * k * d_hat^2 / d * n_ij
    where n_ij = (x_i - x_j) / d,  k_eff = alpha_i * k * d_hat^2.

    Energy Hessian (PSD projection, drops negative eigenvalue components):
        H_i = k_eff / d^2 * (n_ij ⊗ n_ij)

    Mass weighting prevents double-push oscillation for same-color pairs that are
    solved simultaneously:
        alpha_i = m_j / (m_i + m_j)
    Each vertex contributes only its momentum-fraction, so their combined repulsion
    equals one full barrier -- no spurious energy injection, no oscillation.

    Note: recolor-zone pairs (d <= r_vi+r_vj) are also in the watchlist and receive
    the barrier here; because they will be recolored to sequential groups the barrier
    is an extra safety margin only, not the primary safeguard.

    This kernel is launched with dim = |current color group|, processing only the
    vertices in the current Gauss-Seidel color.  Positions come from state_in.particle_q
    which is updated each color via copy_particle_positions_back -- not stale.

    Args:
        particle_ids_in_color: Vertex indices for the current Gauss-Seidel color.
        pos: Current Gauss-Seidel positions (state_in.particle_q).
        particle_mass: Per-vertex mass (0 = pinned particle).
        watchlist_partners: Flat [vi*WATCHLIST_MAX + k] = k-th partner index.
        watchlist_count: Per-vertex number of registered watchlist partners.
        particle_conservative_bounds: OGC scalar bounds r per vertex.
        barrier_stiffness: Stiffness coefficient k.
        particle_forces: Accumulated force array (atomic add).
        particle_hessians: Accumulated Hessian array (atomic add).
    """
    t_id = wp.tid()
    vi = particle_ids_in_color[t_id]

    n_partners = wp.min(watchlist_count[vi], WATCHLIST_MAX_PER_VERTEX)
    if n_partners == 0:
        return

    pos_vi = pos[vi]
    m_vi = particle_mass[vi]
    r_vi = particle_conservative_bounds[vi]

    f_total = wp.vec3(float(0.0), float(0.0), float(0.0))
    h_total = wp.mat33(
        float(0.0), float(0.0), float(0.0),
        float(0.0), float(0.0), float(0.0),
        float(0.0), float(0.0), float(0.0),
    )

    for k in range(n_partners):
        vj = watchlist_partners[vi * WATCHLIST_MAX_PER_VERTEX + k]
        diff = pos_vi - pos[vj]
        dist = wp.length(diff)

        if dist < float(1e-8):
            continue

        r_vj = particle_conservative_bounds[vj]
        d_hat = float(2.0) * (r_vi + r_vj)   # R_watchlist

        if dist >= d_hat or d_hat < float(1e-8):
            continue

        m_vj = particle_mass[vj]
        m_sum = m_vi + m_vj
        if m_sum < float(1e-10):
            continue  # both pinned -- no relative motion possible

        # Mass-weighted stiffness: alpha_i = m_j / (m_i + m_j)
        alpha_i = m_vj / m_sum
        k_eff = alpha_i * barrier_stiffness * d_hat * d_hat

        n_ij = diff / dist

        # Force: -dB/dx_i = k_eff / d * n_ij  (repulsive along n_ij)
        f_i = (k_eff / dist) * n_ij

        # Hessian (PSD rank-1 projection): k_eff / d^2 * (n_ij ⊗ n_ij)
        h_i = (k_eff / (dist * dist)) * wp.outer(n_ij, n_ij)

        f_total = f_total + f_i
        h_total = h_total + h_i

    wp.atomic_add(particle_forces, vi, f_total)
    wp.atomic_add(particle_hessians, vi, h_total)


@wp.kernel
def apply_effective_color_assignments(
    recolored_vertices: wp.array(dtype=wp.int32),
    dynamic_color_ids: wp.array(dtype=wp.int32),
    particle_effective_color: wp.array(dtype=wp.int32),
):
    """Write dynamic color assignments for recolored vertices onto the GPU.

    Args:
        recolored_vertices: Sorted array of vertex indices that need recoloring.
        dynamic_color_ids: New color id for each entry in recolored_vertices.
        particle_effective_color: Output per-vertex effective color array.
    """
    t_id = wp.tid()
    v = recolored_vertices[t_id]
    particle_effective_color[v] = dynamic_color_ids[t_id]
