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

from __future__ import annotations

from newton import data_collector
import time

import warnings

import numpy as np
import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, JointType, Model, State
from ..solver import SolverBase
from .particle_vbd_kernels import (
    NUM_THREADS_PER_COLLISION_PRIMITIVE,
    TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE,
    ParticleForceElementAdjacencyInfo,
    # Topological filtering helper functions
    _set_to_csr,
    accumulate_contact_force_and_hessian,
    accumulate_contact_force_and_hessian_log_collision,
    accumulate_contact_force_and_hessian_no_self_contact,
    accumulate_spring_force_and_hessian,
    build_edge_n_ring_edge_collision_filter,
    build_vertex_n_ring_tris_collision_filter,
    accumulate_same_color_watchlist_barrier_kernel,
    apply_locked_vertex_recovery_kernel,
    apply_planar_truncation_parallel_by_collision,
    apply_truncation_ts,
    compute_displacement,
    build_same_color_watchlist_kernel,
    count_near_penetrations_kernel,
    compute_particle_conservative_bound,
    copy_particle_positions_back,
    # Adjacency building kernels
    count_num_adjacent_edges,
    count_num_adjacent_faces,
    count_num_adjacent_springs,
    fill_adjacent_edges,
    fill_adjacent_faces,
    fill_adjacent_springs,
    # Solver kernels (particle VBD)
    forward_step,
    forward_step_penetration_free,
    solve_trimesh_no_self_contact,
    solve_trimesh_no_self_contact_tile,
    solve_trimesh_with_self_contact_penetration_free,
    solve_trimesh_with_self_contact_penetration_free_tile,
    update_velocity,
)
from .rigid_vbd_kernels import (
    _NUM_CONTACT_THREADS_PER_BODY,
    RigidForceElementAdjacencyInfo,
    # Iteration kernels
    accumulate_body_body_contacts_per_body,  # Body-body (rigid-rigid) contacts (Gauss-Seidel mode)
    accumulate_body_particle_contacts_per_body,  # Body-particle soft contacts (two-way coupling)
    build_body_body_contact_lists,  # Body-body (rigid-rigid) contact adjacency
    build_body_particle_contact_lists,  # Body-particle (rigid-particle) soft-contact adjacency
    compute_cable_dahl_parameters,  # Cable bending plasticity
    copy_rigid_body_transforms_back,
    # Adjacency building kernels
    count_num_adjacent_joints,
    fill_adjacent_joints,
    # Pre-iteration kernels (rigid AVBD)
    forward_step_rigid_bodies,
    solve_rigid_body,
    # Post-iteration kernels
    update_body_velocity,
    update_cable_dahl_state,
    update_duals_body_body_contacts,  # Body-body (rigid-rigid) contacts (AVBD penalty update)
    update_duals_body_particle_contacts,  # Body-particle soft contacts (AVBD penalty update)
    update_duals_joint,  # Cable joints (AVBD penalty update)
    warmstart_body_body_contacts,  # Body-body (rigid-rigid) contacts (penalty warmstart)
    warmstart_body_particle_contacts,  # Body-particle soft contacts (penalty warmstart)
    warmstart_joints,  # Cable joints (stretch & bend)
)
from .tri_mesh_collision import (
    TriMeshCollisionDetector,
    TriMeshCollisionInfo,
)


class SolverVBD(SolverBase):
    """An implicit solver using Vertex Block Descent (VBD) for particles and Augmented VBD (AVBD) for rigid bodies.

    This unified solver supports:
        - Particle simulation (cloth, soft bodies) using the VBD algorithm
        - Rigid body simulation (joints, contacts) using the AVBD algorithm
        - Coupled particle-rigid body systems

    For rigid bodies, the AVBD algorithm uses **soft constraints** with adaptive penalty parameters
    for joints and contacts. Hard constraints are not currently enforced.


    References:
        - Anka He Chen, Ziheng Liu, Yin Yang, and Cem Yuksel. 2024. Vertex Block Descent. ACM Trans. Graph. 43, 4, Article 116 (July 2024), 16 pages.
          https://doi.org/10.1145/3658179
    Note:
        `SolverVBD` requires coloring information for both particles and rigid bodies:

        - Particle coloring: :attr:`newton.Model.particle_color_groups` (required if particles are present)
        - Rigid body coloring: :attr:`newton.Model.body_color_groups` (required if rigid bodies are present)

        Call :meth:`newton.ModelBuilder.color` to automatically color both particles and rigid bodies.

    Example
    -------

    .. code-block:: python

        # Automatically color both particles and rigid bodies
        builder.color()

        model = builder.finalize()

        solver = newton.solvers.SolverVBD(model)

        # Initialize states and contacts
        state_in = model.state()
        state_out = model.state()
        control = model.control()
        contacts = model.collide(state_in)

        # Simulation loop
        for i in range(100):
            contacts = model.collide(state_in)  # Update contacts
            solver.step(state_in, state_out, control, contacts, dt)
            state_in, state_out = state_out, state_in
    """

    def __init__(
        self,
        model: Model,
        # Common parameters
        iterations: int = 10,
        friction_epsilon: float = 1e-2,
        integrate_with_external_rigid_solver: bool = False,
        # Particle parameters
        particle_enable_self_contact: bool = False,
        particle_self_contact_radius: float = 0.2,
        particle_self_contact_margin: float = 0.2,
        particle_conservative_bound_relaxation: float = 0.42,
        particle_vertex_contact_buffer_size: int = 32,
        particle_edge_contact_buffer_size: int = 64,
        particle_collision_detection_interval: int = 0,
        particle_edge_parallel_epsilon: float = 1e-5,
        particle_enable_tile_solve: bool = True,
        particle_topological_contact_filter_threshold: int = 2,
        particle_rest_shape_contact_exclusion_radius: float = 0.0,
        particle_external_vertex_contact_filtering_map: dict | None = None,
        particle_external_edge_contact_filtering_map: dict | None = None,
        # Rigid body parameters
        rigid_avbd_beta: float = 1.0e5,
        rigid_avbd_gamma: float = 0.99,
        rigid_contact_k_start: float = 1.0e2,  # AVBD: initial stiffness for all body contacts (body-body + body-particle)
        rigid_joint_linear_k_start: float = 1.0e4,  # AVBD: initial stiffness seed for linear joint DOFs (e.g., cable stretch)
        rigid_joint_angular_k_start: float = 1.0e1,  # AVBD: initial stiffness seed for angular joint DOFs (e.g., cable bend)
        rigid_body_contact_buffer_size: int = 64,
        rigid_body_particle_contact_buffer_size: int = 256,
        rigid_enable_dahl_friction: bool = False,  # Cable bending plasticity/hysteresis
        rigid_dahl_eps_max: float | wp.array = 0.5,  # Dahl: max persistent strain
        rigid_dahl_tau: float | wp.array = 1.0,  # Dahl: memory decay length

        ogc_contact: bool = False,
        diagnostic_same_color_pairs: bool = False,
        # Watchlist / dynamic recoloring / same-color barrier parameters (Stage 3-5)
        enable_watchlist: bool = False,
        watchlist_buffer_multiplier: int = 4,
        dynamic_recoloring: bool = False,
        enable_same_color_barrier: bool = False,
        same_color_barrier_stiffness: float = 1.0e3,
        same_color_barrier_d_hat: float = -1.0,
        recolor_distance: float = -1.0,
        watchlist_distance: float = -1.0,
        recovery_alpha: float = 0.0,
        recovery_epsilon: float = 1e-4,
        recovery_stiffness: float = 1.0e3,
        recovery_log_path: str = "",
        use_planar_dat: bool = False,
        planar_dat_gamma_r: float = 0.9,
        use_cuda_graph: bool = False,
        diagnostics=None,
    ):
        """
        Args:
            model: The `Model` object used to initialize the integrator. Must be identical to the `Model` object passed
                to the `step` function.

            Common parameters:

            iterations: Number of VBD iterations per step.
            friction_epsilon: Threshold to smooth small relative velocities in friction computation (used for both particle
                and rigid body contacts).

            Particle parameters:

            particle_enable_self_contact: Whether to enable self-contact detection for particles.
            particle_self_contact_radius: The radius used for self-contact detection. This is the distance at which
                vertex-triangle pairs and edge-edge pairs will start to interact with each other.
            particle_self_contact_margin: The margin used for self-contact detection. This is the distance at which
                vertex-triangle pairs and edge-edge will be considered in contact generation. It should be larger than
                `particle_self_contact_radius` to avoid missing contacts.
            integrate_with_external_rigid_solver: Indicator for coupled rigid body-cloth simulation. When set to `True`,
                the solver assumes rigid bodies are integrated by an external solver (one-way coupling).
            particle_conservative_bound_relaxation: Relaxation factor for conservative penetration-free projection.
            particle_vertex_contact_buffer_size: Preallocation size for each vertex's vertex-triangle collision buffer.
            particle_edge_contact_buffer_size: Preallocation size for edge's edge-edge collision buffer.
            particle_collision_detection_interval: Controls how frequently particle self-contact detection is applied
                during the simulation. If set to a value < 0, collision detection is only performed once before the
                initialization step. If set to 0, collision detection is applied twice: once before and once immediately
                after initialization. If set to a value `n` >= 1, collision detection is applied before every `n` VBD
                iterations.
            particle_edge_parallel_epsilon: Threshold to detect near-parallel edges in edge-edge collision handling.
            particle_enable_tile_solve: Whether to accelerate the particle solver using tile API.
            particle_topological_contact_filter_threshold: Maximum topological distance (measured in rings) under which candidate
                self-contacts are discarded. Set to a higher value to tolerate contacts between more closely connected mesh
                elements. Only used when `particle_enable_self_contact` is `True`. Note that setting this to a value larger than 3 will
                result in a significant increase in computation time.
            particle_rest_shape_contact_exclusion_radius: Additional world-space distance threshold for filtering topologically close
                primitives. Candidate contacts with a rest separation shorter than this value are ignored. The distance is
                evaluated in the rest configuration conveyed by `model.particle_q`. Only used when `particle_enable_self_contact` is `True`.
            particle_external_vertex_contact_filtering_map: Optional dictionary used to exclude additional vertex-triangle pairs during
                contact generation. Keys must be vertex primitive ids (integers), and each value must be a `list` or
                `set` containing the triangle primitives to be filtered out. Only used when `particle_enable_self_contact` is `True`.
            particle_external_edge_contact_filtering_map: Optional dictionary used to exclude additional edge-edge pairs during contact
                generation. Keys must be edge primitive ids (integers), and each value must be a `list` or `set`
                containing the edges to be filtered out. Only used when `particle_enable_self_contact` is `True`.

            Rigid body parameters:

            rigid_avbd_beta: Penalty ramp rate for rigid body constraints (how fast k grows with constraint violation).
            rigid_avbd_gamma: Warmstart decay for penalty k (cross-step decay factor for rigid body constraints).
            rigid_contact_k_start: Initial penalty stiffness for all body contact constraints, including both body-body (rigid-rigid)
                and body-particle (rigid-particle) contacts (AVBD).
            rigid_joint_linear_k_start: Initial penalty seed for linear joint DOFs (e.g., cable stretch). Used to seed the per-DOF
                adaptive penalties for all linear joint constraints.
            rigid_joint_angular_k_start: Initial penalty seed for angular joint DOFs (e.g., cable bend). Used to seed the per-DOF
                adaptive penalties for all angular joint constraints.
            rigid_body_contact_buffer_size: Max body-body (rigid-rigid) contacts per rigid body for per-body contact lists (tune based on expected body-body contact density).
            rigid_body_particle_contact_buffer_size: Max body-particle (rigid-particle) contacts per rigid body for per-body soft-contact lists (tune based on expected body-particle contact density).
            rigid_enable_dahl_friction: Enable Dahl hysteresis friction model for cable bending (default: False).
            rigid_dahl_eps_max: Maximum persistent strain (curvature) [rad] for Dahl friction model. Can be:
                - float: Same value for all joints
                - array: Per-joint values for heterogeneous cables
            rigid_dahl_tau: Memory decay length [rad] for Dahl friction model. Controls plasticity. Can be:
                - float: Same value for all joints
                - array: Per-joint values

        Note:
            - The `integrate_with_external_rigid_solver` argument enables one-way coupling between rigid body and soft body
              solvers. If set to True, the rigid states should be integrated externally, with `state_in` passed to `step`
              representing the previous rigid state and `state_out` representing the current one. Frictional forces are
              computed accordingly.
            - `particle_vertex_contact_buffer_size`, `particle_edge_contact_buffer_size`, `rigid_body_contact_buffer_size`,
              and `rigid_body_particle_contact_buffer_size` are fixed and will not be dynamically resized during runtime.
              Setting them too small may result in undetected collisions (particles) or contact overflow (rigid body
              contacts).
              Setting them excessively large may increase memory usage and degrade performance.
            diagnostic_same_color_pairs: When True, prints a per-substep summary of same-color vertex pairs
                found in the existing OGC collision buffers. Reports candidate count, pairs within R_recolor,
                pairs within R_watchlist, and minimum same-color distance. Has no effect on solver behavior.
                Requires particle_enable_self_contact=True (OGC bounds must be available).
            enable_watchlist: When True, builds a GPU-resident flat buffer of same-color vertex pairs in the
                watchlist region (R_recolor < d <= R_watchlist) after each OGC bound computation.
                Pairs in the recolor region (d <= R_recolor) are stored for potential recoloring.
                Does not change solver behavior by itself. Requires particle_enable_self_contact=True.
                This buffer is the foundation for the Stage 5 same-color barrier.
            watchlist_buffer_multiplier: Pre-allocation scale for the watchlist/recolor pair buffers.
                Buffer capacity = watchlist_buffer_multiplier * particle_count pairs.
                If the actual pair count exceeds capacity, excess pairs are silently dropped
                and a warning is printed. Increase this value if overflow warnings appear.
            dynamic_recoloring: When True, dynamically reassigns vertex colors before each substep
                to separate same-color pairs in the recolor region (d <= R_recolor = r[i] + r[j]).
                Automatically enables enable_watchlist=True. After each substep, colors are restored
                to the original static coloring before the next recoloring pass.
                Requires particle_enable_self_contact=True. Keep optional behind this flag.
            enable_same_color_barrier: When True, applies an IPC-like repulsive barrier force
                to same-color watchlist pairs (R_recolor < d <= R_watchlist) that are closer than
                same_color_barrier_d_hat at each VBD iteration. Forces are scaled by inverse-mass
                weights (alpha_i = inv_mass_i/(inv_mass_i+inv_mass_j)) to mitigate double-push.
                Automatically enables enable_watchlist=True.
                Requires particle_enable_self_contact=True.
            same_color_barrier_stiffness: Stiffness κ for the same-color barrier potential.
                Uses the same C2-continuous contact energy as VBD self-contact.
            same_color_barrier_d_hat: Activation distance threshold for the same-color barrier (m).
                Pairs closer than this value receive a repulsive force. Default (-1.0) uses
                particle_self_contact_radius.
            recolor_distance: Fixed vertex-vertex distance threshold for the recolor region (m).
                Same-color pairs with d <= recolor_distance are candidates for dynamic recoloring.
                Default (-1.0) uses particle_self_contact_margin.
            watchlist_distance: Fixed vertex-vertex distance threshold for the watchlist region (m).
                Same-color pairs with recolor_distance < d <= watchlist_distance enter the watchlist.
                Default (-1.0) uses 2 * particle_self_contact_margin.
            recovery_alpha: Step size for the locked-vertex recovery gradient (m / (N/m)).
                Disabled when 0.0 (default). Applied once per substep after finalize_particles.
                Requires particle_enable_self_contact=True.
            recovery_epsilon: OGC bound threshold below which a vertex is considered locked (m).
                Default 1e-4. Should match the red-dot visualization threshold.
            recovery_stiffness: Stiffness κ for the per-contact recovery barrier gradient.
                Default 1e3. Scales the gradient magnitude before alpha is applied.
            recovery_log_path: Path prefix for CSV metric log. If non-empty, writes
                /debug/recovery_metrics_{alpha:.4f}.csv each substep.
                Columns: substep, locked_count, recovery_applied, avg_pen_depth, max_pen_depth.

        """
        super().__init__(model)

        self.ogc_contact = ogc_contact
        if self.ogc_contact:
            print()
            print(">>> OGC Contact mode ON <<<")
            print()

        self.diagnostic_same_color_pairs = diagnostic_same_color_pairs
        if diagnostic_same_color_pairs and not particle_enable_self_contact:
            print(
                "[same-color diag] WARNING: diagnostic_same_color_pairs=True has no effect "
                "without particle_enable_self_contact=True (OGC bounds are not computed)."
            )

        # Both dynamic_recoloring and enable_same_color_barrier require the watchlist buffer.
        # Resolve the final enable_watchlist value before storing it.
        if dynamic_recoloring and not enable_watchlist:
            enable_watchlist = True
        if enable_same_color_barrier and not enable_watchlist:
            enable_watchlist = True

        self.enable_watchlist = enable_watchlist
        if enable_watchlist and not particle_enable_self_contact:
            print(
                "[watchlist] WARNING: enable_watchlist=True has no effect "
                "without particle_enable_self_contact=True (OGC bounds are not computed)."
            )

        self.dynamic_recoloring = dynamic_recoloring
        if dynamic_recoloring and not particle_enable_self_contact:
            print(
                "[recolor] WARNING: dynamic_recoloring=True has no effect "
                "without particle_enable_self_contact=True (OGC bounds are not computed)."
            )

        self.enable_same_color_barrier = enable_same_color_barrier
        self.same_color_barrier_stiffness = same_color_barrier_stiffness
        # Resolve d_hat: if negative, use particle_self_contact_radius (set after _init_particle_system)
        self._same_color_barrier_d_hat_override = same_color_barrier_d_hat
        # Resolve recolor/watchlist fixed thresholds: if negative, use margin-based defaults
        # (set after _init_particle_system once particle_self_contact_margin is known)
        self._recolor_distance_override = recolor_distance
        self._watchlist_distance_override = watchlist_distance

        # Recovery mechanism parameters
        self.recovery_alpha = recovery_alpha
        self.recovery_epsilon = recovery_epsilon
        self.recovery_stiffness = recovery_stiffness
        self._recovery_log_path = recovery_log_path
        self._recovery_substep = 0
        self._recovery_csv_file = None
        self._recovery_csv_writer = None

        # Planar-DAT parameters
        self.use_planar_dat = use_planar_dat
        self.planar_dat_gamma_r = planar_dat_gamma_r

        # CUDA graph for particle solve iterations (one graph per dt value)
        self.use_cuda_graph = use_cuda_graph
        self._particle_solve_graph: "wp.Graph | None" = None
        self._particle_solve_graph_dt: float = 0.0

        # Diagnostics hook (VBDDiagnostics or None)
        self._diag = diagnostics

        if enable_same_color_barrier and not particle_enable_self_contact:
            print(
                "[barrier] WARNING: enable_same_color_barrier=True has no effect "
                "without particle_enable_self_contact=True (OGC bounds are not computed)."
            )

        # Common parameters
        self.iterations = iterations
        data_collector.record_to_scene("iter_per_substep", self.iterations)
        self.friction_epsilon = friction_epsilon

        # Rigid integration mode: when True, rigid bodies are integrated by an external
        # solver (one-way coupling). SolverVBD will not move rigid bodies, but can still
        # participate in particle-rigid interaction on the particle side.
        self.integrate_with_external_rigid_solver = integrate_with_external_rigid_solver

        # Initialize particle system
        self._init_particle_system(
            model,
            particle_enable_self_contact,
            particle_self_contact_radius,
            particle_self_contact_margin,
            particle_conservative_bound_relaxation,
            particle_vertex_contact_buffer_size,
            particle_edge_contact_buffer_size,
            particle_collision_detection_interval,
            particle_edge_parallel_epsilon,
            particle_enable_tile_solve,
            particle_topological_contact_filter_threshold,
            particle_rest_shape_contact_exclusion_radius,
            particle_external_vertex_contact_filtering_map,
            particle_external_edge_contact_filtering_map,
            enable_watchlist,
            watchlist_buffer_multiplier,
            dynamic_recoloring,
        )

        # Resolve fixed recolor/watchlist thresholds first (d_hat default depends on R_recolor)
        margin = particle_self_contact_margin
        if self._recolor_distance_override < 0.0:
            self.recolor_fixed_threshold = margin
        else:
            self.recolor_fixed_threshold = self._recolor_distance_override
        if self._watchlist_distance_override < 0.0:
            self.watchlist_fixed_threshold = 2.0 * margin
        else:
            self.watchlist_fixed_threshold = self._watchlist_distance_override

        # Resolve same-color barrier d_hat: default to R_recolor so barrier fires as pairs
        # approach the recolor zone from the watchlist region during VBD iterations
        if self._same_color_barrier_d_hat_override < 0.0:
            self.same_color_barrier_d_hat = self.recolor_fixed_threshold
        else:
            self.same_color_barrier_d_hat = self._same_color_barrier_d_hat_override

        print(
            f"[watchlist] R_recolor={self.recolor_fixed_threshold:.4f}m  "
            f"R_watchlist={self.watchlist_fixed_threshold:.4f}m  "
            f"d_hat={self.same_color_barrier_d_hat:.4f}m"
        )

        # Initialize rigid body system and rigid-particle (body-particle) interaction state
        self._init_rigid_system(
            model,
            rigid_avbd_beta,
            rigid_avbd_gamma,
            rigid_contact_k_start,
            rigid_joint_linear_k_start,
            rigid_joint_angular_k_start,
            rigid_body_contact_buffer_size,
            rigid_body_particle_contact_buffer_size,
            rigid_enable_dahl_friction,
            rigid_dahl_eps_max,
            rigid_dahl_tau,
        )

        # Rigid-only flag to control whether to update cross-step history
        # (rigid warmstart state such as contact/joint history).
        # Defaults to True. This setting applies only to the next call to :meth:`step` and is then
        # reset to ``True``. This is useful for substepping, where history update frequency might
        # differ from the simulation step frequency (e.g. updating only on the first substep).
        # This flag is automatically reset to True after each step().
        # Rigid warmstart update flag (contacts/joints).
        self.update_rigid_history = True

    def _init_particle_system(
        self,
        model: Model,
        particle_enable_self_contact: bool,
        particle_self_contact_radius: float,
        particle_self_contact_margin: float,
        particle_conservative_bound_relaxation: float,
        particle_vertex_contact_buffer_size: int,
        particle_edge_contact_buffer_size: int,
        particle_collision_detection_interval: int,
        particle_edge_parallel_epsilon: float,
        particle_enable_tile_solve: bool,
        particle_topological_contact_filter_threshold: int,
        particle_rest_shape_contact_exclusion_radius: float,
        particle_external_vertex_contact_filtering_map: dict | None,
        particle_external_edge_contact_filtering_map: dict | None,
        enable_watchlist: bool = False,
        watchlist_buffer_multiplier: int = 4,
        dynamic_recoloring: bool = False,
    ):
        """Initialize particle-specific data structures and settings."""
        # Early exit if no particles
        if model.particle_count == 0:
            return

        self.particle_collision_detection_interval = particle_collision_detection_interval
        self.particle_topological_contact_filter_threshold = particle_topological_contact_filter_threshold
        self.particle_rest_shape_contact_exclusion_radius = particle_rest_shape_contact_exclusion_radius

        # Particle state storage
        self.particle_q_prev = wp.zeros_like(
            model.particle_q, device=self.device
        )  # per-substep previous q (for velocity)
        self.inertia = wp.zeros_like(model.particle_q, device=self.device)  # inertial target positions

        # Particle adjacency info
        self.particle_adjacency = self.compute_particle_force_element_adjacency(model).to(self.device)

        # Self-contact settings
        self.particle_enable_self_contact = particle_enable_self_contact
        self.particle_self_contact_radius = particle_self_contact_radius
        self.particle_self_contact_margin = particle_self_contact_margin
        self.particle_q_rest = model.particle_q

        # Tile solve settings
        if model.device.is_cpu and particle_enable_tile_solve:
            warnings.warn("Tiled solve requires model.device='cuda'. Tiled solve is disabled.", stacklevel=2)

        self.use_particle_tile_solve = particle_enable_tile_solve and model.device.is_cuda

        soft_contact_max = model.shape_count * model.particle_count
        if particle_enable_self_contact:
            if particle_self_contact_margin < particle_self_contact_radius:
                raise ValueError(
                    "particle_self_contact_margin is smaller than particle_self_contact_radius, this will result in missing contacts and cause instability.\n"
                    "It is advisable to make particle_self_contact_margin 1.5-2 times larger than particle_self_contact_radius."
                )

            self.particle_conservative_bound_relaxation = particle_conservative_bound_relaxation
            self.pos_prev_collision_detection = wp.zeros_like(model.particle_q, device=self.device)
            self.particle_conservative_bounds = wp.zeros((model.particle_count,), dtype=float, device=self.device)

            self.trimesh_collision_detector = TriMeshCollisionDetector(
                self.model,
                vertex_collision_buffer_pre_alloc=particle_vertex_contact_buffer_size,
                edge_collision_buffer_pre_alloc=particle_edge_contact_buffer_size,
                edge_edge_parallel_epsilon=particle_edge_parallel_epsilon,
                v_adj_edges=self.particle_adjacency.v_adj_edges,
                v_adj_edges_offsets=self.particle_adjacency.v_adj_edges_offsets,
                v_adj_faces=self.particle_adjacency.v_adj_faces,
                v_adj_faces_offsets=self.particle_adjacency.v_adj_faces_offsets,
                record_triangle_contacting_vertices=self.use_planar_dat,
            )

            self.compute_particle_contact_filtering_list(
                particle_external_vertex_contact_filtering_map, particle_external_edge_contact_filtering_map
            )

            self.trimesh_collision_detector.set_collision_filter_list(
                self.particle_vertex_triangle_contact_filtering_list,
                self.particle_vertex_triangle_contact_filtering_list_offsets,
                self.particle_edge_edge_contact_filtering_list,
                self.particle_edge_edge_contact_filtering_list_offsets,
            )

            self.trimesh_collision_info = wp.array(
                [self.trimesh_collision_detector.collision_info], dtype=TriMeshCollisionInfo, device=self.device
            )

            # Buffers for GitHub-style parallel-by-collision Planar-DAT
            self.truncation_ts = wp.ones(model.particle_count, dtype=float, device=self.device)
            self.particle_displacements = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)

            self.collision_evaluation_kernel_launch_size = max(
                self.model.particle_count * NUM_THREADS_PER_COLLISION_PRIMITIVE,
                self.model.edge_count * NUM_THREADS_PER_COLLISION_PRIMITIVE,
                soft_contact_max,
            )
        else:
            self.collision_evaluation_kernel_launch_size = soft_contact_max

        # Particle force and hessian storage
        self.particle_forces = wp.zeros(self.model.particle_count, dtype=wp.vec3, device=self.device)
        self.stvk_forces = wp.zeros(self.model.particle_count, dtype=wp.vec3, device=self.device)
        self.particle_hessians = wp.zeros(self.model.particle_count, dtype=wp.mat33, device=self.device)

        if data_collector.is_log_collision():
            vt_contact_max = self.model.particle_count * particle_vertex_contact_buffer_size
            ee_contact_max = self.model.particle_count * particle_edge_contact_buffer_size
            self.collision_counter = wp.zeros(1, dtype=int)
            self.all_collision_count = int(soft_contact_max + vt_contact_max + ee_contact_max)
            print(self.all_collision_count, "here")
            self.contacts_index = wp.empty(self.all_collision_count, dtype=int)
            self.contacts_is_self_col = wp.empty(self.all_collision_count, dtype=bool)
            self.contacts_is_body_cloth_col = wp.empty(self.all_collision_count, dtype=bool)
            self.contacts_is_vt_col = wp.empty(self.all_collision_count, dtype=bool)
            self.contacts_is_ee_col = wp.empty(self.all_collision_count, dtype=bool)
            self.contacts_vid = wp.empty(self.all_collision_count, dtype=int)
            self.contacts_fid = wp.empty(self.all_collision_count, dtype=int)
            self.contacts_eid1 = wp.empty(self.all_collision_count, dtype=int)
            self.contacts_eid2 = wp.empty(self.all_collision_count, dtype=int)
            self.contacts_T = wp.empty(self.all_collision_count, dtype=wp.vec3, device=self.device)
            self.contacts_B = wp.empty(self.all_collision_count, dtype=wp.vec3, device=self.device)
            self.contacts_collision_normal = wp.empty(self.all_collision_count, dtype=wp.vec3, device=self.device)
            self.contacts_u_norm = wp.empty(self.all_collision_count, dtype=float, device=self.device)
            self.contacts_eps_u = wp.empty(self.all_collision_count, dtype=float, device=self.device)
            self.contacts_is_slip = wp.empty(self.all_collision_count, dtype=bool, device=self.device)
            self.contacts_friction_force = wp.empty(self.all_collision_count, dtype=wp.vec3, device=self.device)
            self.contacts_normal_contact_force_sum = wp.empty(self.all_collision_count, dtype=wp.vec3, device=self.device)
            self.contacts_normal_contact_force_min = wp.empty(self.all_collision_count, dtype=wp.vec3, device=self.device)


            self.contacts_normal_contact_force0 = wp.empty(self.all_collision_count, dtype=wp.vec3, device=self.device)
            self.contacts_normal_contact_force1 = wp.empty(self.all_collision_count, dtype=wp.vec3, device=self.device)
            self.contacts_normal_contact_force2 = wp.empty(self.all_collision_count, dtype=wp.vec3, device=self.device)
            self.contacts_normal_contact_force3 = wp.empty(self.all_collision_count, dtype=wp.vec3, device=self.device)
            self.contacts_friction0 = wp.empty(self.all_collision_count, dtype=wp.vec3, device=self.device)
            self.contacts_friction1 = wp.empty(self.all_collision_count, dtype=wp.vec3, device=self.device)
            self.contacts_friction2 = wp.empty(self.all_collision_count, dtype=wp.vec3, device=self.device)
            self.contacts_friction3 = wp.empty(self.all_collision_count, dtype=wp.vec3, device=self.device)
            self.contacts_v_list = wp.empty(self.all_collision_count, dtype=wp.vec4i, device=self.device)
            self.contacts_mu = wp.empty(self.all_collision_count, dtype=float, device=self.device)

        # Watchlist/recolor buffers (Stage 3-4): GPU-resident same-color pair lists.
        # Populated by build_same_color_watchlist_kernel after each OGC bound computation.
        # Only allocated when both enable_watchlist and particle_enable_self_contact are True.
        if enable_watchlist and particle_enable_self_contact:
            max_pairs = watchlist_buffer_multiplier * model.particle_count
            self.same_color_watchlist_pairs = wp.zeros(
                2 * max_pairs, dtype=wp.int32, device=self.device
            )
            self.same_color_watchlist_count = wp.zeros(1, dtype=wp.int32, device=self.device)
            self.same_color_watchlist_max_pairs = max_pairs
            # Recolor pairs buffer: pairs with d <= R_recolor that need color splitting
            max_recolor_pairs = max(max_pairs // 2, 1)
            self.same_color_recolor_pairs = wp.zeros(
                2 * max_recolor_pairs, dtype=wp.int32, device=self.device
            )
            self.same_color_recolor_count = wp.zeros(1, dtype=wp.int32, device=self.device)
            self.same_color_recolor_max_pairs = max_recolor_pairs
            # Stage 6 diagnostic buffers: min same-color distance and near-penetration count
            self.same_color_min_dist = wp.array([3.4e38], dtype=float, device=self.device)
            self.near_penetration_count = wp.zeros(1, dtype=wp.int32, device=self.device)

        # Tight penetration counter: vertices with min_dist < 1e-4 at substep start
        if particle_enable_self_contact:
            self._tight_pen_count = wp.zeros(1, dtype=wp.int32, device=self.device)
            self._tight_pen_threshold = 1e-4  # 0.1 mm

        # Recovery mechanism buffers (allocated always; used only when recovery_alpha > 0)
        if particle_enable_self_contact:
            self._recovery_locked_count = wp.zeros(1, dtype=wp.int32, device=self.device)
            self._recovery_applied_count = wp.zeros(1, dtype=wp.int32, device=self.device)
            self._recovery_sum_pen = wp.zeros(1, dtype=float, device=self.device)
            self._recovery_max_pen = wp.zeros(1, dtype=float, device=self.device)
            # Positions of recovered vertices (indexed by recovery_applied_count)
            self._recovery_positions_buf = wp.zeros(
                model.particle_count, dtype=wp.vec3, device=self.device
            )
            # Gradient direction diagnostics
            self._recovery_grad_dir_ok = wp.zeros(1, dtype=wp.int32, device=self.device)
            self._recovery_grad_dir_total = wp.zeros(1, dtype=wp.int32, device=self.device)
            self._recovery_new_pen_count = wp.zeros(1, dtype=wp.int32, device=self.device)
            # Per-vertex diagnostics (indexed by recovery_applied_count)
            self._recovery_vertex_ids = wp.zeros(model.particle_count, dtype=wp.int32, device=self.device)
            self._recovery_d_before = wp.zeros(model.particle_count, dtype=float, device=self.device)
            self._recovery_d_after = wp.zeros(model.particle_count, dtype=float, device=self.device)
            self._recovery_dot_check = wp.zeros(model.particle_count, dtype=float, device=self.device)

        # Dynamic recoloring state (Stage 4): save original coloring for per-substep restore.
        if dynamic_recoloring and particle_enable_self_contact:
            # Save original static coloring so it can be restored before each recoloring pass.
            self._orig_particle_colors_np = model.particle_colors.numpy().copy()
            self._orig_particle_color_groups = [
                g.numpy().copy() for g in model.particle_color_groups
            ]
            # Pre-allocate stable GPU arrays for CUDA-graph-safe restore.
            # _restore_original_colors reuses these objects every substep so the CUDA graph's
            # baked-in GPU pointers remain valid across substeps.
            self._orig_particle_colors_gpu = wp.array(
                self._orig_particle_colors_np, dtype=wp.int32, device=self.device
            )
            self._orig_particle_color_groups_gpu: list[wp.array] = [
                wp.array(g, dtype=wp.int32, device=self.device)
                for g in self._orig_particle_color_groups
            ]
            # Build mesh-edge neighbor adjacency for greedy recoloring.
            # edge_indices[:, 2] and [:, 3] are the two endpoint vertex indices.
            n = model.particle_count
            self._recolor_neighbors: list[set[int]] = [set() for _ in range(n)]
            if model.edge_indices is not None and model.edge_indices.shape[0] > 0:
                edges_np = model.edge_indices.numpy()  # shape (E, >=4)
                for row in edges_np:
                    v0, v1 = int(row[2]), int(row[3])
                    self._recolor_neighbors[v0].add(v1)
                    self._recolor_neighbors[v1].add(v0)

        # Validation
        if len(self.model.particle_color_groups) == 0:
            raise ValueError(
                "model.particle_color_groups is empty! When using the SolverVBD you must call ModelBuilder.color() "
                "or ModelBuilder.set_coloring() before calling ModelBuilder.finalize()."
            )

    def _init_rigid_system(
        self,
        model: Model,
        rigid_avbd_beta: float,
        rigid_avbd_gamma: float,
        rigid_contact_k_start: float,
        rigid_joint_linear_k_start: float,
        rigid_joint_angular_k_start: float,
        rigid_body_contact_buffer_size: int,
        rigid_body_particle_contact_buffer_size: int,
        rigid_enable_dahl_friction: bool,
        rigid_dahl_eps_max: float | wp.array,
        rigid_dahl_tau: float | wp.array,
    ):
        """Initialize rigid body-specific AVBD data structures and settings.

        This includes:
          - Rigid-only AVBD state (joints, body-body contacts, Dahl friction)
          - Shared interaction state for body-particle (rigid-particle) soft contacts
        """
        # AVBD penalty parameters
        self.avbd_beta = rigid_avbd_beta
        self.avbd_gamma = rigid_avbd_gamma

        # Common initial penalty seed / lower bound for body contacts (clamped to non-negative)
        self.k_start_body_contact = max(0.0, rigid_contact_k_start)

        # -------------------------------------------------------------
        # Rigid-only AVBD state (used when SolverVBD integrates bodies)
        # -------------------------------------------------------------
        if not self.integrate_with_external_rigid_solver and model.body_count > 0:
            # State storage
            self.body_q_prev = wp.zeros_like(
                model.body_q, device=self.device
            )  # per-substep previous body pose (for velocity)
            self.body_inertia_q = wp.zeros_like(model.body_q, device=self.device)  # inertial target poses for AVBD

            # Adjacency and dimensions
            self.rigid_adjacency = self.compute_rigid_force_element_adjacency(model).to(self.device)

            # Force accumulation arrays
            self.body_torques = wp.zeros(self.model.body_count, dtype=wp.vec3, device=self.device)
            self.body_forces = wp.zeros(self.model.body_count, dtype=wp.vec3, device=self.device)

            # Hessian blocks (6x6 block structure: angular-angular, angular-linear, linear-linear)
            self.body_hessian_aa = wp.zeros(self.model.body_count, dtype=wp.mat33, device=self.device)
            self.body_hessian_al = wp.zeros(self.model.body_count, dtype=wp.mat33, device=self.device)
            self.body_hessian_ll = wp.zeros(self.model.body_count, dtype=wp.mat33, device=self.device)

            # Per-body contact lists
            # Body-body (rigid-rigid) contact adjacency (CSR-like: per-body counts and flat index array)
            self.body_body_contact_buffer_pre_alloc = rigid_body_contact_buffer_size
            self.body_body_contact_counts = wp.zeros(self.model.body_count, dtype=wp.int32, device=self.device)
            self.body_body_contact_indices = wp.zeros(
                self.model.body_count * self.body_body_contact_buffer_pre_alloc, dtype=wp.int32, device=self.device
            )

            # Body-particle (rigid-particle) contact adjacency (CSR-like: per-body counts and flat index array)
            self.body_particle_contact_buffer_pre_alloc = rigid_body_particle_contact_buffer_size
            self.body_particle_contact_counts = wp.zeros(self.model.body_count, dtype=wp.int32, device=self.device)
            self.body_particle_contact_indices = wp.zeros(
                self.model.body_count * self.body_particle_contact_buffer_pre_alloc,
                dtype=wp.int32,
                device=self.device,
            )

            # AVBD constraint penalties
            # Joint penalties (per-DOF adaptive penalties seeded from joint-wide linear/angular stiffness)
            self.joint_penalty_k = self._init_joint_penalty_k(rigid_joint_linear_k_start, rigid_joint_angular_k_start)

            # Contact penalties (adaptive penalties for body-body contacts)
            if model.shape_count > 0:
                if not hasattr(model, "rigid_contact_max") or model.rigid_contact_max is None:
                    raise ValueError(
                        "Model.rigid_contact_max is not set. Ensure the model was created via ModelBuilder.finalize()."
                    )
                max_contacts = model.rigid_contact_max
                # Per-contact AVBD penalty for body-body contacts
                self.body_body_contact_penalty_k = wp.full(
                    (max_contacts,), self.k_start_body_contact, dtype=float, device=self.device
                )

                # Pre-computed averaged body-body contact material properties (computed once per step in warmstart)
                self.body_body_contact_material_ke = wp.zeros(max_contacts, dtype=float, device=self.device)
                self.body_body_contact_material_kd = wp.zeros(max_contacts, dtype=float, device=self.device)
                self.body_body_contact_material_mu = wp.zeros(max_contacts, dtype=float, device=self.device)

            # Dahl friction model (cable bending plasticity)
            # State variables for Dahl hysteresis (persistent across timesteps)
            self.joint_sigma_prev = wp.zeros(model.joint_count, dtype=wp.vec3, device=self.device)
            self.joint_kappa_prev = wp.zeros(model.joint_count, dtype=wp.vec3, device=self.device)
            self.joint_dkappa_prev = wp.zeros(model.joint_count, dtype=wp.vec3, device=self.device)

            # Pre-computed Dahl parameters (frozen during iterations, updated per timestep)
            self.joint_sigma_start = wp.zeros(model.joint_count, dtype=wp.vec3, device=self.device)
            self.joint_C_fric = wp.zeros(model.joint_count, dtype=wp.vec3, device=self.device)

            # Dahl model configuration
            self.enable_dahl_friction = rigid_enable_dahl_friction
            self.joint_dahl_eps_max = wp.zeros(model.joint_count, dtype=float, device=self.device)
            self.joint_dahl_tau = wp.zeros(model.joint_count, dtype=float, device=self.device)

            if rigid_enable_dahl_friction:
                if model.joint_count == 0:
                    self.enable_dahl_friction = False
                else:
                    self._init_dahl_params(rigid_dahl_eps_max, rigid_dahl_tau, model)

        # -------------------------------------------------------------
        # Body-particle interaction - shared state
        # -------------------------------------------------------------
        # Soft contact penalties (adaptive penalties for body-particle contacts)
        # Use same initial penalty as body-body contacts
        max_soft_contacts = model.shape_count * model.particle_count
        # Per-contact AVBD penalty for body-particle soft contacts (same initial seed as body-body)
        self.body_particle_contact_penalty_k = wp.full(
            (max_soft_contacts,), self.k_start_body_contact, dtype=float, device=self.device
        )

        # Pre-computed averaged body-particle soft contact material properties (computed once per step in warmstart)
        # These correspond to body-particle soft contacts and are averaged between model.soft_contact_*
        # and shape material properties.
        self.body_particle_contact_material_ke = wp.zeros(max_soft_contacts, dtype=float, device=self.device)
        self.body_particle_contact_material_kd = wp.zeros(max_soft_contacts, dtype=float, device=self.device)
        self.body_particle_contact_material_mu = wp.zeros(max_soft_contacts, dtype=float, device=self.device)

        # Validation
        has_bodies = self.model.body_count > 0
        has_body_coloring = len(self.model.body_color_groups) > 0

        if has_bodies and not has_body_coloring:
            raise ValueError(
                "model.body_color_groups is empty but rigid bodies are present! When using the SolverVBD you must call ModelBuilder.color() "
                "or ModelBuilder.set_coloring() before calling ModelBuilder.finalize()."
            )

    # =====================================================
    # Initialization Helper Methods
    # =====================================================

    def _init_joint_penalty_k(self, k_start_joint_linear: float, k_start_joint_angular: float):
        """
        Build initial per-DOF joint penalty array on CPU and upload to solver device.
        - Default seed is a global rigid-body penalty for all DOFs.
        - Optionally override cable stretch/bend DOFs per joint.
        """
        dof_count = self.model.joint_dof_count
        with wp.ScopedDevice("cpu"):
            # Seed all DOFs with the joint-linear stiffness and specialize cable bend DOFs below.
            # This keeps a single pair of joint-wide seeds as the authoritative source of joint stiffness.
            stretch_k = max(0.0, k_start_joint_linear)
            joint_k_min_np = np.full((dof_count,), 0.0, dtype=float)
            joint_k0_np = np.full((dof_count,), stretch_k, dtype=float)

            jt_cpu = self.model.joint_type.to("cpu")
            jdofs_cpu = self.model.joint_qd_start.to("cpu")

            jt = jt_cpu.numpy() if hasattr(jt_cpu, "numpy") else np.asarray(jt_cpu, dtype=int)
            jdofs = jdofs_cpu.numpy() if hasattr(jdofs_cpu, "numpy") else np.asarray(jdofs_cpu, dtype=int)

            n_j = self.model.joint_count
            bend_k = max(0.0, k_start_joint_angular)
            for j in range(n_j):
                if jt[j] == JointType.CABLE:
                    dof0 = jdofs[j]
                    # DOF 0: cable stretch; DOF 1: cable bend
                    joint_k0_np[dof0] = stretch_k
                    joint_k0_np[dof0 + 1] = bend_k
                    # Per-DOF lower bounds: use k_start_* for cable stretch/bend, 0 otherwise
                    joint_k_min_np[dof0] = stretch_k
                    joint_k_min_np[dof0 + 1] = bend_k

            # Upload to device: initial penalties and per-DOF lower bounds
            joint_penalty_k_min = wp.array(joint_k_min_np, dtype=float, device=self.device)
            self.joint_penalty_k_min = joint_penalty_k_min
            return wp.array(joint_k0_np, dtype=float, device=self.device)

    def _init_dahl_params(self, eps_max_input, tau_input, model):
        """
        Initialize per-joint Dahl friction parameters.

        Args:
            eps_max_input: float or array-like. Maximum strain (curvature) [rad].
                - Scalar: broadcast to all joints
                - Array-like (length = model.joint_count): per-joint values
                - Per-joint disable: set value to 0 for that joint
            tau_input: float or array-like. Memory decay length [rad].
                - Scalar: broadcast to all joints
                - Array-like (length = model.joint_count): per-joint values
                - Per-joint disable: set value to 0 for that joint
            model: Model object

        Notes:
            - This function validates shapes and converts to device arrays; it does not clamp or validate ranges.
              Kernels perform any necessary early-outs based on zero values.
            - To disable Dahl friction:
                - Globally: pass enable_dahl_friction=False to the constructor
                - Per-joint: set dahl_eps_max=0 or dahl_tau=0 for those joints
        """
        n = model.joint_count

        # eps_max
        if isinstance(eps_max_input, (int, float)):
            self.joint_dahl_eps_max = wp.full(n, eps_max_input, dtype=float, device=self.device)
        else:
            # Convert to numpy first
            x = eps_max_input.to("cpu") if hasattr(eps_max_input, "to") else eps_max_input
            eps_np = x.numpy() if hasattr(x, "numpy") else np.asarray(x, dtype=float)
            if eps_np.shape[0] != n:
                raise ValueError(f"dahl_eps_max length {eps_np.shape[0]} != joint_count {n}")
            # Direct host-to-device copy
            self.joint_dahl_eps_max = wp.array(eps_np, dtype=float, device=self.device)

        # tau
        if isinstance(tau_input, (int, float)):
            self.joint_dahl_tau = wp.full(n, tau_input, dtype=float, device=self.device)
        else:
            # Convert to numpy first
            x = tau_input.to("cpu") if hasattr(tau_input, "to") else tau_input
            tau_np = x.numpy() if hasattr(x, "numpy") else np.asarray(x, dtype=float)
            if tau_np.shape[0] != n:
                raise ValueError(f"dahl_tau length {tau_np.shape[0]} != joint_count {n}")
            # Direct host-to-device copy
            self.joint_dahl_tau = wp.array(tau_np, dtype=float, device=self.device)

    # =====================================================
    # Adjacency Building Methods
    # =====================================================

    def compute_particle_force_element_adjacency(self, model):
        adjacency = ParticleForceElementAdjacencyInfo()
        edges_array = model.edge_indices.to("cpu")
        spring_array = model.spring_indices.to("cpu")
        face_indices = model.tri_indices.to("cpu")

        with wp.ScopedDevice("cpu"):
            if edges_array.size:
                # Build vertex-edge adjacency data.
                num_vertex_adjacent_edges = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)

                wp.launch(
                    kernel=count_num_adjacent_edges,
                    inputs=[edges_array, num_vertex_adjacent_edges],
                    dim=1,
                )

                num_vertex_adjacent_edges = num_vertex_adjacent_edges.numpy()
                vertex_adjacent_edges_offsets = np.empty(shape=(self.model.particle_count + 1,), dtype=wp.int32)
                vertex_adjacent_edges_offsets[1:] = np.cumsum(2 * num_vertex_adjacent_edges)[:]
                vertex_adjacent_edges_offsets[0] = 0
                adjacency.v_adj_edges_offsets = wp.array(vertex_adjacent_edges_offsets, dtype=wp.int32)

                # Temporal variables to record how much adjacent edges has been filled to each vertex.
                vertex_adjacent_edges_fill_count = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)

                edge_adjacency_array_size = 2 * num_vertex_adjacent_edges.sum()
                # vertex order: o0: 0, o1: 1, v0: 2, v1: 3,
                adjacency.v_adj_edges = wp.empty(shape=(edge_adjacency_array_size,), dtype=wp.int32)

                wp.launch(
                    kernel=fill_adjacent_edges,
                    inputs=[
                        edges_array,
                        adjacency.v_adj_edges_offsets,
                        vertex_adjacent_edges_fill_count,
                        adjacency.v_adj_edges,
                    ],
                    dim=1,
                )
            else:
                adjacency.v_adj_edges_offsets = wp.empty(shape=(0,), dtype=wp.int32)
                adjacency.v_adj_edges = wp.empty(shape=(0,), dtype=wp.int32)

            if face_indices.size:
                # Compute adjacent triangles.
                # Count number of adjacent faces for each vertex.
                num_vertex_adjacent_faces = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)
                wp.launch(kernel=count_num_adjacent_faces, inputs=[face_indices, num_vertex_adjacent_faces], dim=1)

                # Preallocate memory based on counting results.
                num_vertex_adjacent_faces = num_vertex_adjacent_faces.numpy()
                vertex_adjacent_faces_offsets = np.empty(shape=(self.model.particle_count + 1,), dtype=wp.int32)
                vertex_adjacent_faces_offsets[1:] = np.cumsum(2 * num_vertex_adjacent_faces)[:]
                vertex_adjacent_faces_offsets[0] = 0
                adjacency.v_adj_faces_offsets = wp.array(vertex_adjacent_faces_offsets, dtype=wp.int32)

                vertex_adjacent_faces_fill_count = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)

                face_adjacency_array_size = 2 * num_vertex_adjacent_faces.sum()
                # (face, vertex_order) * num_adj_faces * num_particles
                # vertex order: v0: 0, v1: 1, o0: 2, v2: 3
                adjacency.v_adj_faces = wp.empty(shape=(face_adjacency_array_size,), dtype=wp.int32)

                wp.launch(
                    kernel=fill_adjacent_faces,
                    inputs=[
                        face_indices,
                        adjacency.v_adj_faces_offsets,
                        vertex_adjacent_faces_fill_count,
                        adjacency.v_adj_faces,
                    ],
                    dim=1,
                )
            else:
                adjacency.v_adj_faces_offsets = wp.empty(shape=(0,), dtype=wp.int32)
                adjacency.v_adj_faces = wp.empty(shape=(0,), dtype=wp.int32)

            if spring_array.size:
                # Build vertex-springs adjacency data.
                num_vertex_adjacent_spring = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)

                wp.launch(
                    kernel=count_num_adjacent_springs,
                    inputs=[spring_array, num_vertex_adjacent_spring],
                    dim=1,
                )

                num_vertex_adjacent_spring = num_vertex_adjacent_spring.numpy()
                vertex_adjacent_springs_offsets = np.empty(shape=(self.model.particle_count + 1,), dtype=wp.int32)
                vertex_adjacent_springs_offsets[1:] = np.cumsum(num_vertex_adjacent_spring)[:]
                vertex_adjacent_springs_offsets[0] = 0
                adjacency.v_adj_springs_offsets = wp.array(vertex_adjacent_springs_offsets, dtype=wp.int32)

                # Temporal variables to record how much adjacent springs has been filled to each vertex.
                vertex_adjacent_springs_fill_count = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)
                adjacency.v_adj_springs = wp.empty(shape=(num_vertex_adjacent_spring.sum(),), dtype=wp.int32)

                wp.launch(
                    kernel=fill_adjacent_springs,
                    inputs=[
                        spring_array,
                        adjacency.v_adj_springs_offsets,
                        vertex_adjacent_springs_fill_count,
                        adjacency.v_adj_springs,
                    ],
                    dim=1,
                )

            else:
                adjacency.v_adj_springs_offsets = wp.empty(shape=(0,), dtype=wp.int32)
                adjacency.v_adj_springs = wp.empty(shape=(0,), dtype=wp.int32)

        return adjacency

    def compute_particle_contact_filtering_list(
        self, external_vertex_contact_filtering_map, external_edge_contact_filtering_map
    ):
        if self.model.tri_count:
            v_tri_filter_sets = None
            edge_edge_filter_sets = None
            if self.particle_topological_contact_filter_threshold >= 2:
                if self.particle_adjacency.v_adj_faces_offsets.size > 0:
                    v_tri_filter_sets = build_vertex_n_ring_tris_collision_filter(
                        self.particle_topological_contact_filter_threshold,
                        self.model.particle_count,
                        self.model.edge_indices.numpy(),
                        self.particle_adjacency.v_adj_edges.numpy(),
                        self.particle_adjacency.v_adj_edges_offsets.numpy(),
                        self.particle_adjacency.v_adj_faces.numpy(),
                        self.particle_adjacency.v_adj_faces_offsets.numpy(),
                    )
                if self.particle_adjacency.v_adj_edges_offsets.size > 0:
                    edge_edge_filter_sets = build_edge_n_ring_edge_collision_filter(
                        self.particle_topological_contact_filter_threshold,
                        self.model.edge_indices.numpy(),
                        self.particle_adjacency.v_adj_edges.numpy(),
                        self.particle_adjacency.v_adj_edges_offsets.numpy(),
                    )

            if external_vertex_contact_filtering_map is not None:
                if v_tri_filter_sets is None:
                    v_tri_filter_sets = [set() for _ in range(self.model.particle_count)]
                for vertex_id, filter_set in external_vertex_contact_filtering_map.items():
                    v_tri_filter_sets[vertex_id].update(filter_set)

            if external_edge_contact_filtering_map is not None:
                if edge_edge_filter_sets is None:
                    edge_edge_filter_sets = [set() for _ in range(self.model.edge_indices.shape[0])]
                for edge_id, filter_set in external_edge_contact_filtering_map.items():
                    edge_edge_filter_sets[edge_id].update(filter_set)

            if v_tri_filter_sets is None:
                self.particle_vertex_triangle_contact_filtering_list = None
                self.particle_vertex_triangle_contact_filtering_list_offsets = None
            else:
                (
                    self.particle_vertex_triangle_contact_filtering_list,
                    self.particle_vertex_triangle_contact_filtering_list_offsets,
                ) = _set_to_csr(v_tri_filter_sets)
                self.particle_vertex_triangle_contact_filtering_list = wp.array(
                    self.particle_vertex_triangle_contact_filtering_list, dtype=int, device=self.device
                )
                self.particle_vertex_triangle_contact_filtering_list_offsets = wp.array(
                    self.particle_vertex_triangle_contact_filtering_list_offsets, dtype=int, device=self.device
                )

            if edge_edge_filter_sets is None:
                self.particle_edge_edge_contact_filtering_list = None
                self.particle_edge_edge_contact_filtering_list_offsets = None
            else:
                (
                    self.particle_edge_edge_contact_filtering_list,
                    self.particle_edge_edge_contact_filtering_list_offsets,
                ) = _set_to_csr(edge_edge_filter_sets)
                self.particle_edge_edge_contact_filtering_list = wp.array(
                    self.particle_edge_edge_contact_filtering_list, dtype=int, device=self.device
                )
                self.particle_edge_edge_contact_filtering_list_offsets = wp.array(
                    self.particle_edge_edge_contact_filtering_list_offsets, dtype=int, device=self.device
                )

    def compute_rigid_force_element_adjacency(self, model):
        """
        Build CSR adjacency between rigid bodies and joints.

        Returns an instance of RigidForceElementAdjacencyInfo with:
          - body_adj_joints: flattened joint ids
          - body_adj_joints_offsets: CSR offsets of size body_count + 1

        Notes:
            - Runs on CPU to avoid GPU atomics; kernels iterate serially over joints (dim=1).
            - When there are no joints, offsets are an all-zero array of length body_count + 1.
        """
        adjacency = RigidForceElementAdjacencyInfo()

        with wp.ScopedDevice("cpu"):
            # Build body-joint adjacency data (rigid-only)
            if model.joint_count > 0:
                joint_parent_cpu = model.joint_parent.to("cpu")
                joint_child_cpu = model.joint_child.to("cpu")

                num_body_adjacent_joints = wp.zeros(shape=(model.body_count,), dtype=wp.int32)
                wp.launch(
                    kernel=count_num_adjacent_joints,
                    inputs=[joint_parent_cpu, joint_child_cpu, num_body_adjacent_joints],
                    dim=1,
                )

                num_body_adjacent_joints = num_body_adjacent_joints.numpy()
                body_adjacent_joints_offsets = np.empty(shape=(model.body_count + 1,), dtype=wp.int32)
                body_adjacent_joints_offsets[1:] = np.cumsum(num_body_adjacent_joints)[:]
                body_adjacent_joints_offsets[0] = 0
                adjacency.body_adj_joints_offsets = wp.array(body_adjacent_joints_offsets, dtype=wp.int32)

                body_adjacent_joints_fill_count = wp.zeros(shape=(model.body_count,), dtype=wp.int32)
                adjacency.body_adj_joints = wp.empty(shape=(num_body_adjacent_joints.sum(),), dtype=wp.int32)

                wp.launch(
                    kernel=fill_adjacent_joints,
                    inputs=[
                        joint_parent_cpu,
                        joint_child_cpu,
                        adjacency.body_adj_joints_offsets,
                        body_adjacent_joints_fill_count,
                        adjacency.body_adj_joints,
                    ],
                    dim=1,
                )
            else:
                # No joints: create offset array of zeros (size body_count + 1) so indexing works
                adjacency.body_adj_joints_offsets = wp.zeros(shape=(model.body_count + 1,), dtype=wp.int32)
                adjacency.body_adj_joints = wp.empty(shape=(0,), dtype=wp.int32)

        return adjacency

    # =====================================================
    # Main Solver Methods
    # =====================================================

    def set_rigid_history_update(self, update: bool):
        """Set whether the next step() should update rigid solver history (warmstarts).

        This setting applies only to the next call to :meth:`step` and is then reset to ``True``.
        This is useful for substepping, where history update frequency might differ from the
        simulation step frequency (e.g. updating only on the first substep).

        Args:
            update: If True, update rigid warmstart state. If False, reuse previous.
        """
        self.update_rigid_history = update

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts,
        dt: float,
    ):
        """Execute one simulation timestep using VBD (particles) and AVBD (rigid bodies).

        The solver follows a 3-phase structure:
        1. Initialize: Forward integrate particles and rigid bodies, detect collisions, warmstart penalties
        2. Iterate: Interleave particle VBD iterations and rigid body AVBD iterations
        3. Finalize: Update velocities and persistent state (Dahl friction)

        To control rigid substepping behavior (warmstart history), call
        :meth:`set_rigid_history_update`
        before calling this method. It defaults to ``True`` and is reset to ``True`` after each call.

        Args:
            state_in: Input state.
            state_out: Output state.
            control: Control inputs.
            contacts: Collision contacts.
            dt: Time step size.
        """
        # Use and reset the rigid history update flag (warmstarts).
        update_rigid_history = self.update_rigid_history
        self.update_rigid_history = True

        if self._diag is not None:
            self._diag.on_before_prediction(state_in, self, dt)

        self.initialize_rigid_bodies(state_in, contacts, dt, update_rigid_history)
        self.initialize_particles(state_in, dt)

        if self._diag is not None:
            self._diag.on_after_prediction(state_in, self, dt)

        for iter_num in range(self.iterations):
            self.solve_rigid_body_iteration(state_in, state_out, contacts, dt)
            self.solve_particle_iteration(state_in, state_out, contacts, dt, iter_num, self.iterations-1)
            if self._diag is not None and self._diag.log_iterations:
                self._diag.on_after_solver_iteration(state_in, self, dt, iter_num)

        if self._diag is not None:
            self._diag.on_after_solver(state_in, self, dt)

        self.finalize_rigid_bodies(state_out, dt)
        self.finalize_particles(state_out, dt)

        if self.recovery_alpha > 0.0 and self.particle_enable_self_contact:
            self._apply_locked_vertex_recovery(state_out)

        if self._diag is not None:
            self._diag.on_end_of_step(state_out, self, dt)

    def initialize_particles(self, state_in: State, dt: float):
        """Initialize particle positions for the VBD iteration."""
        model = self.model

        # Early exit if no particles
        if model.particle_count == 0:
            return

        if self.particle_enable_self_contact:
            # Dynamic recoloring: restore original static coloring before collision detection
            # so that the watchlist kernel sees canonical same-color pairs, not stale recolored ones.
            if self.dynamic_recoloring:
                self._restore_original_colors()

            # Collision detection before initialization to compute conservative bounds
            if data_collector.is_log_nothing():
                self.collision_detection_penetration_free(state_in, -1)
            else:
                self.collision_detection_penetration_free_log_collision(state_in, -1)

            if self.diagnostic_same_color_pairs:
                self._diagnose_same_color_pairs(state_in)

            # Dynamic recoloring: apply greedy reassignment for recolor-region pairs
            if self.dynamic_recoloring:
                self._apply_dynamic_recoloring_from_gpu()

            wp.launch(
                kernel=forward_step_penetration_free,
                inputs=[
                    dt,
                    model.gravity,
                    self.particle_q_prev,
                    state_in.particle_q,
                    state_in.particle_qd,
                    model.particle_inv_mass,
                    state_in.particle_f,
                    model.particle_flags,
                    self.pos_prev_collision_detection,
                    self.particle_conservative_bounds,
                    self.inertia,
                ],
                dim=model.particle_count,
                device=self.device,
            )

            # Algorithm 3, step 4: apply Planar-DAT to the initial inertia guess.
            # self.inertia holds the untruncated inertia positions (= X + ΔX_init).
            # state_in.particle_q is overwritten with the truncated result.
            # GitHub 2-pass: parallel-by-collision truncation ratio computation.
            if self._diag is not None:
                self._diag.on_before_truncation(self.inertia, state_in, self)
            if self.use_planar_dat and self.particle_enable_self_contact:
                # Pass 1a: displacement = inertia - X_0 (per vertex)
                wp.launch(
                    kernel=compute_displacement,
                    dim=model.particle_count,
                    inputs=[self.inertia, self.pos_prev_collision_detection],
                    outputs=[self.particle_displacements],
                    device=self.device,
                )
                # Pass 1b: for each collision pair, compute division plane and
                #          atomic-min each vertex's truncation ratio into truncation_ts.
                self.truncation_ts.fill_(1.0)
                wp.launch(
                    kernel=apply_planar_truncation_parallel_by_collision,
                    dim=self.collision_evaluation_kernel_launch_size,
                    inputs=[
                        self.pos_prev_collision_detection,
                        self.particle_displacements,
                        model.tri_indices,
                        model.edge_indices,
                        self.trimesh_collision_info,
                        self.trimesh_collision_detector.edge_edge_parallel_epsilon,
                        self.planar_dat_gamma_r,
                        self.truncation_ts,
                    ],
                    device=self.device,
                )
                # Pass 2: apply per-vertex truncation ratio + isotropic displacement cap.
                max_disp = 0.5 * self.planar_dat_gamma_r * 1.5 * self.particle_self_contact_radius
                wp.launch(
                    kernel=apply_truncation_ts,
                    dim=model.particle_count,
                    inputs=[
                        self.pos_prev_collision_detection,
                        self.particle_displacements,
                        self.truncation_ts,
                        max_disp,
                    ],
                    outputs=[state_in.particle_q],
                    device=self.device,
                )
            if self._diag is not None:
                self._diag.on_after_truncation(self.inertia, state_in, self)
        else:
            wp.launch(
                kernel=forward_step,
                inputs=[
                    dt,
                    model.gravity,
                    self.particle_q_prev,
                    state_in.particle_q,
                    state_in.particle_qd,
                    model.particle_inv_mass,
                    state_in.particle_f,
                    model.particle_flags,
                    self.inertia,
                ],
                dim=model.particle_count,
                device=self.device,
            )

    def initialize_rigid_bodies(
        self,
        state_in: State,
        contacts: Contacts,
        dt: float,
        update_rigid_history: bool,
    ):
        """Initialize rigid body states for AVBD solver (pre-iteration phase).

        Performs forward integration, builds contact lists, and warmstarts AVBD penalty parameters.
        """
        model = self.model

        # ---------------------------
        # Rigid-only initialization
        # ---------------------------
        if model.body_count > 0 and not self.integrate_with_external_rigid_solver:
            # Forward integrate rigid bodies
            wp.launch(
                kernel=forward_step_rigid_bodies,
                inputs=[
                    dt,
                    model.gravity,
                    state_in.body_f,
                    model.body_com,
                    model.body_inertia,
                    model.body_inv_mass,
                    model.body_inv_inertia,
                    state_in.body_q,  # input/output
                    state_in.body_qd,  # input/output
                ],
                outputs=[
                    self.body_q_prev,
                    self.body_inertia_q,
                ],
                dim=model.body_count,
                device=self.device,
            )

            if update_rigid_history:
                # Use the Contacts buffer capacity as launch dimension
                contact_launch_dim = contacts.rigid_contact_max

                # Build per-body contact lists once per step
                # Build body-body (rigid-rigid) contact lists
                self.body_body_contact_counts.zero_()
                wp.launch(
                    kernel=build_body_body_contact_lists,
                    dim=contact_launch_dim,
                    inputs=[
                        contacts.rigid_contact_count,
                        contacts.rigid_contact_shape0,
                        contacts.rigid_contact_shape1,
                        model.shape_body,
                        self.body_body_contact_buffer_pre_alloc,
                    ],
                    outputs=[
                        self.body_body_contact_counts,
                        self.body_body_contact_indices,
                    ],
                    device=self.device,
                )

                # Warmstart AVBD body-body contact penalties and pre-compute material properties
                wp.launch(
                    kernel=warmstart_body_body_contacts,
                    inputs=[
                        contacts.rigid_contact_count,
                        contacts.rigid_contact_shape0,
                        contacts.rigid_contact_shape1,
                        model.shape_material_ke,
                        model.shape_material_kd,
                        model.shape_material_mu,
                        self.k_start_body_contact,
                    ],
                    outputs=[
                        self.body_body_contact_penalty_k,
                        self.body_body_contact_material_ke,
                        self.body_body_contact_material_kd,
                        self.body_body_contact_material_mu,
                    ],
                    dim=contact_launch_dim,
                    device=self.device,
                )

                # Warmstart AVBD penalty parameters for joints using the same cadence
                # as contact history updates.
                if model.joint_count > 0:
                    wp.launch(
                        kernel=warmstart_joints,
                        inputs=[
                            model.joint_target_ke,
                            self.joint_penalty_k_min,
                            self.avbd_gamma,
                            self.joint_penalty_k,  # input/output
                        ],
                        dim=model.joint_dof_count,
                        device=self.device,
                    )

            # Compute Dahl hysteresis parameters for cable bending (once per timestep, frozen during iterations)
            if self.enable_dahl_friction and model.joint_count > 0:
                wp.launch(
                    kernel=compute_cable_dahl_parameters,
                    inputs=[
                        model.joint_type,
                        model.joint_parent,
                        model.joint_child,
                        model.joint_X_p,
                        model.joint_X_c,
                        model.joint_qd_start,
                        model.joint_target_ke,
                        self.body_q_prev,  # Use previous body transforms (start of step) for linearization
                        model.body_q,  # rest body transforms
                        model.body_com,
                        self.joint_sigma_prev,
                        self.joint_kappa_prev,
                        self.joint_dkappa_prev,
                        self.joint_dahl_eps_max,
                        self.joint_dahl_tau,
                    ],
                    outputs=[
                        self.joint_sigma_start,
                        self.joint_C_fric,
                    ],
                    dim=model.joint_count,
                    device=self.device,
                )

        # ---------------------------
        # Body-particle interaction
        # ---------------------------
        if model.particle_count > 0 and update_rigid_history:
            # Build body-particle (rigid-particle) contact lists only when SolverVBD
            # is integrating rigid bodies itself; the external rigid solver path
            # does not use these per-body adjacency structures. Also skip if there
            # are no rigid bodies in the model.
            if not self.integrate_with_external_rigid_solver and model.body_count > 0:
                self.body_particle_contact_counts.zero_()
                wp.launch(
                    kernel=build_body_particle_contact_lists,
                    dim=contacts.soft_contact_max,
                    inputs=[
                        contacts.soft_contact_count,
                        contacts.soft_contact_shape,
                        model.shape_body,
                        self.body_particle_contact_buffer_pre_alloc,
                    ],
                    outputs=[
                        self.body_particle_contact_counts,
                        self.body_particle_contact_indices,
                    ],
                    device=self.device,
                )

            # Warmstart AVBD body-particle contact penalties and pre-compute material properties.
            # This is useful both when SolverVBD integrates rigid bodies and when an external
            # rigid solver is used, since cloth-rigid soft contacts still rely on these penalties.
            soft_contact_launch_dim = contacts.soft_contact_max
            wp.launch(
                kernel=warmstart_body_particle_contacts,
                inputs=[
                    contacts.soft_contact_count,
                    contacts.soft_contact_shape,
                    model.soft_contact_ke,
                    model.soft_contact_kd,
                    model.soft_contact_mu,
                    model.shape_material_ke,
                    model.shape_material_kd,
                    model.shape_material_mu,
                    self.k_start_body_contact,
                ],
                outputs=[
                    self.body_particle_contact_penalty_k,
                    self.body_particle_contact_material_ke,
                    self.body_particle_contact_material_kd,
                    self.body_particle_contact_material_mu,
                ],
                dim=soft_contact_launch_dim,
                device=self.device,
            )

    def _build_particle_solve_graph(
        self, state_in: State, state_out: State, contacts: Contacts, dt: float
    ) -> "wp.Graph":
        """Capture one particle VBD iteration (kernel work only, no CD) as a reusable CUDA graph.

        During capture ``wp.get_stream(self.device).is_capturing`` is ``True``, so
        ``solve_particle_iteration`` automatically skips all CPU-side work (collision
        detection, timing, logging) and records only the GPU kernel launches.
        """
        wp.capture_begin(device=self.device, force_module_load=True)
        # iter_num=0 / max_iter_num=-1: collision-detection interval check fires for
        # interval==0, but the is_capturing guard inside solve_particle_iteration
        # skips it anyway.  Logging is off (data_collector.is_log_nothing()==True
        # is a precondition enforced by the caller), so no .numpy() sync occurs.
        self.solve_particle_iteration(state_in, state_out, contacts, dt, iter_num=0, max_iter_num=-1)
        return wp.capture_end(device=self.device)

    def solve_particle_iteration(self, state_in: State, state_out: State, contacts: Contacts, dt: float, iter_num: int, max_iter_num=-1):
        """Solve one VBD iteration for particles."""
        model = self.model

        # Select rigid-body poses for particle-rigid contact evaluation
        if self.integrate_with_external_rigid_solver:
            body_q_for_particles = state_out.body_q
            body_q_prev_for_particles = state_in.body_q
            body_qd_for_particles = state_out.body_qd
        else:
            body_q_for_particles = state_in.body_q
            if model.body_count > 0:
                body_q_prev_for_particles = self.body_q_prev
            else:
                body_q_prev_for_particles = None
            body_qd_for_particles = state_in.body_qd

        # Early exit if no particles
        if model.particle_count == 0:
            return

        # CD and timing: skip during CUDA graph capture (no CPU-GPU sync allowed)
        if not wp.get_stream(self.device).is_capturing:
            # Update collision detection if needed (penetration-free mode only)
            if self.particle_enable_self_contact:
                if (self.particle_collision_detection_interval == 0 and iter_num == 0) or (
                    self.particle_collision_detection_interval >= 1
                    and iter_num % self.particle_collision_detection_interval == 0
                ):
                    if data_collector.is_log_nothing():
                        self.collision_detection_penetration_free(state_in, iter_num)
                    else:
                        col_detect_time_start = time.perf_counter()
                        self.collision_detection_penetration_free_log_collision(state_in, iter_num)
                        col_detect_time_end = time.perf_counter()
                        data_collector.record_to_frame("col_detect_time", col_detect_time_end - col_detect_time_start)
                elif not data_collector.is_log_nothing():
                    data_collector.record_to_frame("col_detect_time", 0)
            elif not data_collector.is_log_nothing():
                data_collector.record_to_frame("col_detect_time", 0)

            # CUDA graph fast path: CD already done above; launch graph for kernel work.
            # Disabled when per-iteration diagnostics are active (requires CPU sync each iter).
            if (
                self.use_cuda_graph
                and data_collector.is_log_nothing()
                and not self.enable_same_color_barrier
                and (self._diag is None or not self._diag.log_iterations)
            ):
                if self._particle_solve_graph is None or dt != self._particle_solve_graph_dt:
                    self._particle_solve_graph = self._build_particle_solve_graph(
                        state_in, state_out, contacts, dt
                    )
                    self._particle_solve_graph_dt = dt
                wp.capture_launch(self._particle_solve_graph)
                return

        # Zero out forces and hessians
        self.particle_forces.zero_()
        self.particle_hessians.zero_()
        self.stvk_forces.zero_()
        if data_collector.is_log_collision():
            self.collision_counter.zero_()

        # Same-color barrier: accumulate watchlist pair forces before color group solves
        if self.enable_same_color_barrier and self.enable_watchlist:
            self._launch_same_color_watchlist_barrier(state_in)

        # Iterate over color groups
        for color in range(len(model.particle_color_groups)):
            # Accumulate contact forces
            if self.particle_enable_self_contact:
                if contacts is not None:
                    if data_collector.is_log_collision() and iter_num == max_iter_num:
                        wp.launch(
                            kernel=accumulate_contact_force_and_hessian_log_collision,
                            dim=self.collision_evaluation_kernel_launch_size,
                            inputs=[
                                dt,
                                color,
                                self.particle_q_prev,
                                state_in.particle_q,
                                model.particle_colors,
                                model.tri_indices,
                                model.edge_indices,
                                # self-contact
                                self.trimesh_collision_info,
                                self.particle_self_contact_radius,
                                model.soft_contact_ke,
                                model.soft_contact_kd,
                                model.soft_contact_mu, # also friction mu
                                self.friction_epsilon,
                                self.trimesh_collision_detector.edge_edge_parallel_epsilon,
                                # body-particle contact
                                model.particle_radius,
                                contacts.soft_contact_particle,
                                contacts.soft_contact_count,
                                contacts.soft_contact_max,
                                self.body_particle_contact_penalty_k,
                                self.body_particle_contact_material_kd,
                                self.body_particle_contact_material_mu,
                                model.shape_material_mu,
                                model.shape_body,
                                body_q_for_particles,
                                body_q_prev_for_particles,
                                body_qd_for_particles,
                                model.body_com,
                                contacts.soft_contact_shape,
                                contacts.soft_contact_body_pos,
                                contacts.soft_contact_body_vel,
                                contacts.soft_contact_normal,

                                self.collision_counter,
                            ],
                            outputs=[
                                self.particle_forces,
                                self.particle_hessians,

                                self.contacts_index,
                                self.contacts_is_self_col,
                                self.contacts_is_body_cloth_col,
                                self.contacts_is_vt_col,
                                self.contacts_is_ee_col,
                                self.contacts_vid,
                                self.contacts_fid,
                                self.contacts_eid1,
                                self.contacts_eid2,
                                self.contacts_T,
                                self.contacts_B,
                                self.contacts_collision_normal,
                                self.contacts_u_norm,
                                self.contacts_eps_u,
                                self.contacts_is_slip,
                                self.contacts_friction_force,
                                self.contacts_normal_contact_force_sum,
                                self.contacts_normal_contact_force_min,

                                self.contacts_normal_contact_force0,
                                self.contacts_normal_contact_force1,
                                self.contacts_normal_contact_force2,
                                self.contacts_normal_contact_force3,
                                self.contacts_friction0,
                                self.contacts_friction1,
                                self.contacts_friction2,
                                self.contacts_friction3,
                                self.contacts_v_list,
                                self.contacts_mu
                            ],
                            device=self.device,
                            max_blocks=model.device.sm_count,
                        )
                    else:
                        wp.launch(
                            kernel=accumulate_contact_force_and_hessian,
                            dim=self.collision_evaluation_kernel_launch_size,
                            inputs=[
                                dt,
                                color,
                                self.particle_q_prev,
                                state_in.particle_q,
                                model.particle_colors,
                                model.tri_indices,
                                model.edge_indices,
                                # self-contact
                                self.trimesh_collision_info,
                                self.particle_self_contact_radius,
                                model.soft_contact_ke,
                                model.soft_contact_kd,
                                model.soft_contact_mu,
                                self.friction_epsilon,
                                self.trimesh_collision_detector.edge_edge_parallel_epsilon,
                                # body-particle contact
                                model.particle_radius,
                                contacts.soft_contact_particle,
                                contacts.soft_contact_count,
                                contacts.soft_contact_max,
                                self.body_particle_contact_penalty_k,
                                self.body_particle_contact_material_kd,
                                self.body_particle_contact_material_mu,
                                model.shape_material_mu,
                                model.shape_body,
                                body_q_for_particles,
                                body_q_prev_for_particles,
                                body_qd_for_particles,
                                model.body_com,
                                contacts.soft_contact_shape,
                                contacts.soft_contact_body_pos,
                                contacts.soft_contact_body_vel,
                                contacts.soft_contact_normal,
                            ],
                            outputs=[
                                self.particle_forces,
                                self.particle_hessians,
                            ],
                            device=self.device,
                            max_blocks=model.device.sm_count,
                        )
            else:
                wp.launch(
                    kernel=accumulate_contact_force_and_hessian_no_self_contact,
                    dim=self.collision_evaluation_kernel_launch_size,
                    inputs=[
                        dt,
                        color,
                        self.particle_q_prev,
                        state_in.particle_q,
                        model.particle_colors,
                        # body-particle contact
                        self.friction_epsilon,
                        model.particle_radius,
                        contacts.soft_contact_particle,
                        contacts.soft_contact_count,
                        contacts.soft_contact_max,
                        self.body_particle_contact_penalty_k,
                        self.body_particle_contact_material_kd,
                        self.body_particle_contact_material_mu,
                        model.shape_material_mu,
                        model.shape_body,
                        body_q_for_particles,
                        body_q_prev_for_particles,
                        state_in.body_qd,
                        model.body_com,
                        contacts.soft_contact_shape,
                        contacts.soft_contact_body_pos,
                        contacts.soft_contact_body_vel,
                        contacts.soft_contact_normal,
                    ],
                    outputs=[
                        self.particle_forces,
                        self.particle_hessians,
                    ],
                    device=self.device,
                )

            # Accumulate spring forces
            if model.spring_count:
                wp.launch(
                    kernel=accumulate_spring_force_and_hessian,
                    inputs=[
                        dt,
                        self.particle_q_prev,
                        state_in.particle_q,
                        model.particle_color_groups[color],
                        self.particle_adjacency,
                        model.spring_indices,
                        model.spring_rest_length,
                        model.spring_stiffness,
                        model.spring_damping,
                    ],
                    outputs=[
                        self.particle_forces,
                        self.particle_hessians,
                    ],
                    dim=model.particle_color_groups[color].size,
                    device=self.device,
                )
            

            # Solve for this color group
            if self.particle_enable_self_contact:
                if self.use_particle_tile_solve:
                    wp.launch(
                        kernel=solve_trimesh_with_self_contact_penetration_free_tile,
                        dim=model.particle_color_groups[color].size * TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE,
                        block_dim=TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE,
                        inputs=[
                            dt,
                            model.particle_color_groups[color],
                            self.particle_q_prev,
                            state_in.particle_q,
                            state_in.particle_qd,
                            model.particle_mass,
                            self.inertia,
                            model.particle_flags,
                            model.tri_indices,
                            model.tri_poses,
                            model.tri_materials,
                            model.tri_areas,
                            model.edge_indices,
                            model.edge_rest_angle,
                            model.edge_rest_length,
                            model.edge_bending_properties,
                            self.particle_adjacency,
                            self.particle_forces,
                            self.particle_hessians,
                            self.pos_prev_collision_detection,
                            self.particle_conservative_bounds,
                            self.use_planar_dat,
                            self.trimesh_collision_detector.collision_info,
                            self.planar_dat_gamma_r,
                            1.5 * self.particle_self_contact_radius,
                            self.trimesh_collision_detector.edge_edge_parallel_epsilon,
                        ],
                        outputs=[
                            state_out.particle_q,
                            self.stvk_forces
                        ],
                        device=self.device,
                    )
                else:
                    wp.launch(
                        kernel=solve_trimesh_with_self_contact_penetration_free,
                        dim=model.particle_color_groups[color].size,
                        inputs=[
                            dt,
                            model.particle_color_groups[color],
                            self.particle_q_prev,
                            state_in.particle_q,
                            state_in.particle_qd,
                            model.particle_mass,
                            self.inertia,
                            model.particle_flags,
                            model.tri_indices,
                            model.tri_poses,
                            model.tri_materials,
                            model.tri_areas,
                            model.edge_indices,
                            model.edge_rest_angle,
                            model.edge_rest_length,
                            model.edge_bending_properties,
                            self.particle_adjacency,
                            self.particle_forces,
                            self.particle_hessians,
                            self.pos_prev_collision_detection,
                            self.particle_conservative_bounds,
                            self.use_planar_dat,
                            self.trimesh_collision_detector.collision_info,
                            self.planar_dat_gamma_r,
                            1.5 * self.particle_self_contact_radius,
                            self.trimesh_collision_detector.edge_edge_parallel_epsilon,
                        ],
                        outputs=[
                            state_out.particle_q,
                            self.stvk_forces
                        ],
                        device=self.device,
                    )
            else:
                if self.use_particle_tile_solve:
                    wp.launch(
                        kernel=solve_trimesh_no_self_contact_tile,
                        inputs=[
                            dt,
                            model.particle_color_groups[color],
                            self.particle_q_prev,
                            state_in.particle_q,
                            state_in.particle_qd,
                            model.particle_mass,
                            self.inertia,
                            model.particle_flags,
                            model.tri_indices,
                            model.tri_poses,
                            model.tri_materials,
                            model.tri_areas,
                            model.edge_indices,
                            model.edge_rest_angle,
                            model.edge_rest_length,
                            model.edge_bending_properties,
                            self.particle_adjacency,
                            self.particle_forces,
                            self.particle_hessians,
                        ],
                        outputs=[
                            state_out.particle_q,
                            self.stvk_forces
                        ],
                        dim=model.particle_color_groups[color].size * TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE,
                        block_dim=TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE,
                        device=self.device,
                    )
                else:
                    wp.launch(
                        kernel=solve_trimesh_no_self_contact,
                        inputs=[
                            dt,
                            model.particle_color_groups[color],
                            self.particle_q_prev,
                            state_in.particle_q,
                            state_in.particle_qd,
                            model.particle_mass,
                            self.inertia,
                            model.particle_flags,
                            model.tri_indices,
                            model.tri_poses,
                            model.tri_materials,
                            model.tri_areas,
                            model.edge_indices,
                            model.edge_rest_angle,
                            model.edge_rest_length,
                            model.edge_bending_properties,
                            self.particle_adjacency,
                            self.particle_forces,
                            self.particle_hessians,
                        ],
                        outputs=[
                            state_out.particle_q,
                            self.stvk_forces
                        ],
                        dim=model.particle_color_groups[color].size,
                        device=self.device,
                    )

            # Copy positions back
            wp.launch(
                kernel=copy_particle_positions_back,
                inputs=[model.particle_color_groups[color], state_in.particle_q, state_out.particle_q],
                dim=model.particle_color_groups[color].size,
                device=self.device,
            )
        # end color loop

        if not data_collector.is_log_nothing():
            data_collector.frame_timer.stop()
            total_force = (self.particle_forces + self.stvk_forces).numpy().reshape(-1, 3)
            force_magnitudes = np.linalg.norm(total_force, axis=1)
            mean_force = np.mean(force_magnitudes)
            data_collector.record_to_iteration("force_residual", mean_force, iter_num)
            
            if data_collector.is_log_collision() and iter_num == max_iter_num:
                valid_size = self.collision_counter.numpy()[0]
                data_collector.record_to_collision(
                    iter_num, 
                    self.contacts_index.numpy()[:valid_size],
                    self.contacts_is_self_col.numpy()[:valid_size],
                    self.contacts_is_body_cloth_col.numpy()[:valid_size],
                    self.contacts_is_vt_col.numpy()[:valid_size],
                    self.contacts_is_ee_col.numpy()[:valid_size],
                    self.contacts_vid.numpy()[:valid_size],
                    self.contacts_fid.numpy()[:valid_size],
                    self.contacts_eid1.numpy()[:valid_size],
                    self.contacts_eid2.numpy()[:valid_size],
                    self.contacts_T.numpy()[:valid_size],
                    self.contacts_B.numpy()[:valid_size],
                    self.contacts_collision_normal.numpy()[:valid_size],
                    self.contacts_u_norm.numpy()[:valid_size],
                    self.contacts_eps_u.numpy()[:valid_size],
                    self.contacts_is_slip.numpy()[:valid_size],
                    self.contacts_normal_contact_force_sum.numpy()[:valid_size],
                    self.contacts_normal_contact_force_min.numpy()[:valid_size],
                    self.contacts_friction_force.numpy()[:valid_size],
                    self.contacts_v_list.numpy()[:valid_size],
                    self.contacts_normal_contact_force0.numpy()[:valid_size],
                    self.contacts_normal_contact_force1.numpy()[:valid_size],
                    self.contacts_normal_contact_force2.numpy()[:valid_size],
                    self.contacts_normal_contact_force3.numpy()[:valid_size],
                    self.contacts_friction0.numpy()[:valid_size],
                    self.contacts_friction1.numpy()[:valid_size],
                    self.contacts_friction2.numpy()[:valid_size],
                    self.contacts_friction3.numpy()[:valid_size],
                    self.contacts_mu.numpy()[:valid_size],
                )
            data_collector.frame_timer.start()
        # print(self.contacts_T1, "<- t1")

    def solve_rigid_body_iteration(self, state_in: State, state_out: State, contacts: Contacts, dt: float):
        """Solve one AVBD iteration for rigid bodies (per-iteration phase).

        Accumulates contact and joint forces/hessians, solves 6x6 rigid body systems per color,
        and updates AVBD penalty parameters (dual update).
        """
        model = self.model

        # Early exit if no rigid bodies
        if model.body_count == 0:
            return

        # If rigid bodies are integrated by an external solver, skip the AVBD rigid-body
        # solve but still update body-particle soft-contact penalties so that adaptive
        # AVBD stiffness is used for cloth-rigid interaction.
        if self.integrate_with_external_rigid_solver:
            if model.particle_count > 0:
                soft_contact_launch_dim = contacts.soft_contact_max
                wp.launch(
                    kernel=update_duals_body_particle_contacts,
                    dim=soft_contact_launch_dim,
                    inputs=[
                        contacts.soft_contact_count,
                        contacts.soft_contact_particle,
                        contacts.soft_contact_shape,
                        contacts.soft_contact_body_pos,
                        contacts.soft_contact_normal,
                        state_in.particle_q,
                        model.particle_radius,
                        model.shape_body,
                        # Rigid poses come from the external solver when
                        # integrate_with_external_rigid_solver=True
                        state_out.body_q,
                        self.body_particle_contact_material_ke,
                        self.avbd_beta,
                        self.body_particle_contact_penalty_k,  # input/output
                    ],
                    device=self.device,
                )
            return

        # Zero out forces and hessianss
        self.body_torques.zero_()
        self.body_forces.zero_()
        self.body_hessian_aa.zero_()
        self.body_hessian_al.zero_()
        self.body_hessian_ll.zero_()

        # AVBD stiffness arrays (adaptive penalties)
        contact_stiffness_array = self.body_body_contact_penalty_k
        joint_stiffness_array = self.joint_penalty_k

        # Use the Contacts buffer capacity as launch dimension
        contact_launch_dim = contacts.rigid_contact_max
        body_color_groups = model.body_color_groups

        # Gauss-Seidel-style per-color updates
        for color in range(len(body_color_groups)):
            color_group = body_color_groups[color]

            # Gauss-Seidel contact accumulation: evaluate contacts for bodies in this color
            # Accumulate body-particle forces and Hessians on bodies (per-body, per-color)
            if model.particle_count > 0:
                wp.launch(
                    kernel=accumulate_body_particle_contacts_per_body,
                    dim=color_group.size * _NUM_CONTACT_THREADS_PER_BODY,
                    inputs=[
                        dt,
                        color_group,
                        # particle state
                        state_in.particle_q,
                        self.particle_q_prev,
                        model.particle_radius,
                        # rigid body state
                        self.body_q_prev,
                        state_in.body_q,
                        model.body_qd,
                        model.body_com,
                        model.body_inv_mass,
                        # AVBD body-particle soft contact penalties and material properties
                        self.friction_epsilon,
                        self.body_particle_contact_penalty_k,
                        self.body_particle_contact_material_kd,
                        self.body_particle_contact_material_mu,
                        # soft contact data (body-particle contacts)
                        contacts.soft_contact_count,
                        contacts.soft_contact_particle,
                        contacts.soft_contact_shape,
                        contacts.soft_contact_body_pos,
                        contacts.soft_contact_body_vel,
                        contacts.soft_contact_normal,
                        # shape/material data
                        model.shape_material_mu,
                        model.shape_body,
                        # per-body adjacency (body-particle contacts)
                        self.body_particle_contact_buffer_pre_alloc,
                        self.body_particle_contact_counts,
                        self.body_particle_contact_indices,
                    ],
                    outputs=[
                        self.body_forces,
                        self.body_torques,
                        self.body_hessian_ll,
                        self.body_hessian_al,
                        self.body_hessian_aa,
                    ],
                    device=self.device,
                )

            # Accumulate body-body (rigid-rigid) contact forces and Hessians on bodies (per-body, per-color)
            wp.launch(
                kernel=accumulate_body_body_contacts_per_body,
                dim=color_group.size * _NUM_CONTACT_THREADS_PER_BODY,
                inputs=[
                    dt,
                    color_group,
                    self.body_q_prev,
                    state_in.body_q,
                    model.body_com,
                    model.body_inv_mass,
                    self.friction_epsilon,
                    contact_stiffness_array,
                    self.body_body_contact_material_kd,
                    self.body_body_contact_material_mu,
                    contacts.rigid_contact_count,
                    contacts.rigid_contact_shape0,
                    contacts.rigid_contact_shape1,
                    contacts.rigid_contact_point0,
                    contacts.rigid_contact_point1,
                    contacts.rigid_contact_normal,
                    contacts.rigid_contact_thickness0,
                    contacts.rigid_contact_thickness1,
                    model.shape_body,
                    self.body_body_contact_buffer_pre_alloc,
                    self.body_body_contact_counts,
                    self.body_body_contact_indices,
                ],
                outputs=[
                    self.body_forces,
                    self.body_torques,
                    self.body_hessian_ll,
                    self.body_hessian_al,
                    self.body_hessian_aa,
                ],
                device=self.device,
            )

            wp.launch(
                kernel=solve_rigid_body,
                inputs=[
                    dt,
                    color_group,
                    state_in.body_q,
                    self.body_q_prev,
                    model.body_q,
                    model.body_mass,
                    model.body_inv_mass,
                    model.body_inertia,
                    self.body_inertia_q,
                    model.body_com,
                    self.rigid_adjacency,
                    model.joint_type,
                    model.joint_parent,
                    model.joint_child,
                    model.joint_X_p,
                    model.joint_X_c,
                    model.joint_qd_start,
                    model.joint_target_kd,
                    joint_stiffness_array,
                    self.joint_sigma_start,
                    self.joint_C_fric,
                    self.body_forces,
                    self.body_torques,
                    self.body_hessian_ll,
                    self.body_hessian_al,
                    self.body_hessian_aa,
                ],
                outputs=[
                    state_out.body_q,
                ],
                dim=color_group.size,
                device=self.device,
            )

            wp.launch(
                kernel=copy_rigid_body_transforms_back,
                inputs=[color_group, state_out.body_q],
                outputs=[state_in.body_q],
                dim=color_group.size,
                device=self.device,
            )

        # AVBD dual update: update adaptive penalties based on constraint violation
        # Update body-body (rigid-rigid) contact penalties
        wp.launch(
            kernel=update_duals_body_body_contacts,
            dim=contact_launch_dim,
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_thickness0,
                contacts.rigid_contact_thickness1,
                model.shape_body,
                state_out.body_q,
                self.body_body_contact_material_ke,
                self.avbd_beta,
                self.body_body_contact_penalty_k,  # input/output
            ],
            device=self.device,
        )

        # Update body-particle contact penalties
        if model.particle_count > 0:
            soft_contact_launch_dim = contacts.soft_contact_max
            wp.launch(
                kernel=update_duals_body_particle_contacts,
                dim=soft_contact_launch_dim,
                inputs=[
                    contacts.soft_contact_count,
                    contacts.soft_contact_particle,
                    contacts.soft_contact_shape,
                    contacts.soft_contact_body_pos,
                    contacts.soft_contact_normal,
                    state_in.particle_q,
                    model.particle_radius,
                    model.shape_body,
                    # Rigid poses come from SolverVBD itself when
                    # integrate_with_external_rigid_solver=False
                    state_in.body_q,
                    self.body_particle_contact_material_ke,
                    self.avbd_beta,
                    self.body_particle_contact_penalty_k,  # input/output
                ],
                device=self.device,
            )

        # Update joint penalties at new positions
        wp.launch(
            kernel=update_duals_joint,
            dim=model.joint_count,
            inputs=[
                model.joint_type,
                model.joint_parent,
                model.joint_child,
                model.joint_X_p,
                model.joint_X_c,
                model.joint_qd_start,
                model.joint_dof_dim,
                model.joint_target_ke,
                state_out.body_q,
                model.body_q,
                model.body_com,
                self.avbd_beta,
                self.joint_penalty_k,  # input/output
            ],
            device=self.device,
        )

    def finalize_particles(self, state_out: State, dt: float):
        """Finalize particle velocities after VBD iterations."""
        # Early exit if no particles
        if self.model.particle_count == 0:
            return

        wp.launch(
            kernel=update_velocity,
            inputs=[dt, self.particle_q_prev, state_out.particle_q, state_out.particle_qd],
            dim=self.model.particle_count,
            device=self.device,
        )

    def finalize_rigid_bodies(self, state_out: State, dt: float):
        """Finalize rigid body velocities and Dahl friction state after AVBD iterations (post-iteration phase).

        Updates rigid body velocities using BDF1 and updates Dahl hysteresis state for cable bending.
        """
        model = self.model

        # Early exit if no rigid bodies or rigid bodies are driven by an external solver
        if model.body_count == 0 or self.integrate_with_external_rigid_solver:
            return

        # Velocity update (BDF1) after all iterations
        wp.launch(
            kernel=update_body_velocity,
            inputs=[dt, state_out.body_q, self.body_q_prev, model.body_com],
            outputs=[state_out.body_qd],
            dim=model.body_count,
            device=self.device,
        )

        # Update Dahl hysteresis state after solver convergence (for next timestep's memory)
        if self.enable_dahl_friction and model.joint_count > 0:
            wp.launch(
                kernel=update_cable_dahl_state,
                inputs=[
                    model.joint_type,
                    model.joint_parent,
                    model.joint_child,
                    model.joint_X_p,
                    model.joint_X_c,
                    model.joint_qd_start,
                    model.joint_target_ke,
                    state_out.body_q,
                    model.body_q,
                    model.body_com,
                    self.joint_dahl_eps_max,
                    self.joint_dahl_tau,
                ],
                outputs=[
                    self.joint_sigma_prev,  # input/output
                    self.joint_kappa_prev,  # input/output
                    self.joint_dkappa_prev,  # input/output
                ],
                dim=model.joint_count,
                device=self.device,
            )

    # called on init(1 time) and solve(iteration times)
    def collision_detection_penetration_free(self, current_state: State, iter_num=-2):
        self.trimesh_collision_detector.refit(current_state.particle_q)
        if self.ogc_contact:
            self.trimesh_collision_detector.vertex_triangle_collision_detection_ogc(
                self.particle_self_contact_margin,
                min_query_radius=self.particle_rest_shape_contact_exclusion_radius,
                min_distance_filtering_ref_pos=self.particle_q_rest,
            )
            self.trimesh_collision_detector.edge_edge_collision_detection_ogc(
                self.particle_self_contact_margin,
                min_query_radius=self.particle_rest_shape_contact_exclusion_radius,
                min_distance_filtering_ref_pos=self.particle_q_rest,
            )
        else:
            self.trimesh_collision_detector.vertex_triangle_collision_detection(
                self.particle_self_contact_margin,
                min_query_radius=self.particle_rest_shape_contact_exclusion_radius,
                min_distance_filtering_ref_pos=self.particle_q_rest,
            )
            self.trimesh_collision_detector.edge_edge_collision_detection(
                self.particle_self_contact_margin,
                min_query_radius=self.particle_rest_shape_contact_exclusion_radius,
                min_distance_filtering_ref_pos=self.particle_q_rest,
            )

        self.pos_prev_collision_detection.assign(current_state.particle_q)
        wp.launch(
            kernel=compute_particle_conservative_bound,
            inputs=[
                self.particle_conservative_bound_relaxation,
                self.particle_self_contact_margin,
                self.particle_adjacency,
                self.trimesh_collision_detector.collision_info,
            ],
            outputs=[
                self.particle_conservative_bounds,
            ],
            dim=self.model.particle_count,
            device=self.device,
        )

        # Count vertices within tight proximity threshold (penetration indicator)
        self._tight_pen_count.zero_()
        wp.launch(
            kernel=count_near_penetrations_kernel,
            inputs=[
                self.trimesh_collision_detector.collision_info,
                self._tight_pen_threshold,
            ],
            outputs=[self._tight_pen_count],
            dim=self.model.particle_count,
            device=self.device,
        )

        if self.enable_watchlist:
            self._run_watchlist_kernel(current_state)


    # called on init(1 time) and solve(iteration times)
    def collision_detection_penetration_free_log_collision(self, current_state: State, iter_num=-2):
        self.trimesh_collision_detector.refit(current_state.particle_q)
        self.trimesh_collision_detector.vertex_triangle_collision_detection(
            self.particle_self_contact_margin,
            min_query_radius=self.particle_rest_shape_contact_exclusion_radius,
            min_distance_filtering_ref_pos=self.particle_q_rest,
        )
        data_collector.record_to_iteration("cloth_self_vt_col_count", self.trimesh_collision_detector.vertex_colliding_triangles_count.numpy().sum(), iter_num)
        self.trimesh_collision_detector.edge_edge_collision_detection(
            self.particle_self_contact_margin,
            min_query_radius=self.particle_rest_shape_contact_exclusion_radius,
            min_distance_filtering_ref_pos=self.particle_q_rest,
        )
        data_collector.record_to_iteration("cloth_self_ee_col_count", self.trimesh_collision_detector.edge_colliding_edges_count.numpy().sum(), iter_num)

        self.pos_prev_collision_detection.assign(current_state.particle_q)
        wp.launch(
            kernel=compute_particle_conservative_bound,
            inputs=[
                self.particle_conservative_bound_relaxation,
                self.particle_self_contact_margin,
                self.particle_adjacency,
                self.trimesh_collision_detector.collision_info,
            ],
            outputs=[
                self.particle_conservative_bounds,
            ],
            dim=self.model.particle_count,
            device=self.device,
        )

        if self.enable_watchlist:
            self._run_watchlist_kernel(current_state)

    def rebuild_bvh(self, state: State):
        """This function will rebuild the BVHs used for detecting self-contacts using the input `state`.

        When the simulated object deforms significantly, simply refitting the BVH can lead to deterioration of the BVH's
        quality. In these cases, rebuilding the entire tree is necessary to achieve better querying efficiency.

        Args:
            state (newton.State):  The state whose particle positions (:attr:`State.particle_q`) will be used for rebuilding the BVHs.
        """
        if self.particle_enable_self_contact:
            self.trimesh_collision_detector.rebuild(state.particle_q)

    def _run_watchlist_kernel(self, current_state: State) -> None:
        """Zero watchlist/recolor counters, launch build_same_color_watchlist_kernel, print diagnostics.

        Called after compute_particle_conservative_bound whenever enable_watchlist=True.
        After return:
          - same_color_watchlist_pairs: (i,j) pairs in the watchlist region (R_recolor < d <= R_watchlist)
          - same_color_recolor_pairs:   (i,j) pairs in the recolor region   (d <= R_recolor)
          - same_color_min_dist:        minimum vertex-vertex distance over all detected same-color pairs
          - near_penetration_count:     number of vertices with nearest-triangle distance < contact_radius
        """
        self.same_color_watchlist_count.zero_()
        self.same_color_recolor_count.zero_()
        self.near_penetration_count.zero_()
        # Reset min_dist sentinel to large value before atomic_min reduction
        self.same_color_min_dist = wp.array([3.4e38], dtype=float, device=self.device)

        wp.launch(
            kernel=build_same_color_watchlist_kernel,
            inputs=[
                current_state.particle_q,
                self.model.particle_colors,
                self.model.tri_indices,
                self.trimesh_collision_detector.collision_info,
                self.recolor_fixed_threshold,
                self.watchlist_fixed_threshold,
                self.same_color_watchlist_max_pairs,
                self.same_color_recolor_max_pairs,
            ],
            outputs=[
                self.same_color_watchlist_pairs,
                self.same_color_watchlist_count,
                self.same_color_recolor_pairs,
                self.same_color_recolor_count,
                self.same_color_min_dist,
            ],
            dim=self.model.particle_count,
            device=self.device,
        )

        # Stage 6: count vertices near penetration (d < contact_radius to nearest triangle)
        wp.launch(
            kernel=count_near_penetrations_kernel,
            inputs=[
                self.trimesh_collision_detector.collision_info,
                self.particle_self_contact_radius,
            ],
            outputs=[
                self.near_penetration_count,
            ],
            dim=self.model.particle_count,
            device=self.device,
        )

        wl = int(self.same_color_watchlist_count.numpy()[0])
        rc = int(self.same_color_recolor_count.numpy()[0])
        min_d_raw = float(self.same_color_min_dist.numpy()[0])
        near_pen = int(self.near_penetration_count.numpy()[0])

        wl_overflow = wl > self.same_color_watchlist_max_pairs
        rc_overflow = rc > self.same_color_recolor_max_pairs
        overflow_str = (" WL_OVERFLOW" if wl_overflow else "") + (" RC_OVERFLOW" if rc_overflow else "")
        min_d_str = f"{min_d_raw:.6f}" if min_d_raw < 3.0e38 else "N/A"
        print(
            f"[watchlist] recolor={min(rc, self.same_color_recolor_max_pairs)}/{self.same_color_recolor_max_pairs} "
            f"watchlist={min(wl, self.same_color_watchlist_max_pairs)}/{self.same_color_watchlist_max_pairs} "
            f"min_sc_dist={min_d_str} "
            f"near_pen={near_pen}"
            f"{overflow_str}"
        )

    def _diagnose_same_color_pairs(self, current_state: State) -> None:
        """CPU-side diagnostic: count same-color vertex pairs using existing collision buffers.

        Categorizes same-color proximate pairs into recolor / watchlist / ignore regions
        using fixed distance thresholds:

            R_recolor   = self.recolor_fixed_threshold
            R_watchlist = self.watchlist_fixed_threshold

        Does not modify any solver state. Runs on CPU; intended for debugging only.
        """
        if not self.particle_enable_self_contact:
            print(
                "[same-color diag] WARNING: particle_enable_self_contact=False. "
                "Collision buffers are not populated. Skipping."
            )
            return

        # --- Download GPU data to CPU numpy ---
        particle_q = current_state.particle_q.numpy()          # (N, 3)
        particle_colors = self.model.particle_colors.numpy()   # (N,)  int
        tri_indices = self.model.tri_indices.numpy()           # (T, 3) int

        detector = self.trimesh_collision_detector
        vt_count = detector.vertex_colliding_triangles_count.numpy()        # (N,)
        vt_buf_sizes = detector.vertex_colliding_triangles_buffer_sizes.numpy()  # (N,)
        vt_offsets = detector.vertex_colliding_triangles_offsets.numpy()    # (N+1,)
        vt_data = detector.vertex_colliding_triangles.numpy()               # flat int32

        R_recolor = self.recolor_fixed_threshold
        R_watchlist = self.watchlist_fixed_threshold

        n_candidate = 0
        n_recolor = 0
        n_watchlist = 0
        min_same_color_dist = float("inf")
        seen: set[tuple[int, int]] = set()

        for i in range(self.model.particle_count):
            color_i = int(particle_colors[i])
            # cap at the actual buffer size to skip overflow slots
            count_i = int(min(vt_count[i], vt_buf_sizes[i]))
            off_i = int(vt_offsets[i])

            for k in range(count_i):
                tri_idx = int(vt_data[2 * (off_i + k) + 1])
                if tri_idx < 0:
                    continue
                # walk the three vertices of the colliding triangle
                for v_order in range(3):
                    j = int(tri_indices[tri_idx, v_order])
                    if j <= i:  # canonical order: only consider (i < j) pairs
                        continue
                    if int(particle_colors[j]) != color_i:
                        continue
                    pair = (i, j)
                    if pair in seen:
                        continue
                    seen.add(pair)

                    d = float(np.linalg.norm(particle_q[i] - particle_q[j]))

                    n_candidate += 1
                    if d < min_same_color_dist:
                        min_same_color_dist = d
                    if d <= R_recolor:
                        n_recolor += 1
                    elif d <= R_watchlist:
                        n_watchlist += 1

        min_dist_str = f"{min_same_color_dist:.6f}" if n_candidate > 0 else "N/A"
        print(
            f"[same-color diag] candidate={n_candidate} "
            f"recolor={n_recolor} "
            f"watchlist={n_watchlist} "
            f"min_dist={min_dist_str}"
        )

    # ------------------------------------------------------------------
    # Stage 4: Dynamic recoloring helpers
    # ------------------------------------------------------------------

    def _restore_original_colors(self) -> None:
        """Restore particle_colors and particle_color_groups to the saved static coloring.

        Reuses pre-allocated GPU arrays (_orig_particle_colors_gpu /
        _orig_particle_color_groups_gpu) so that the CUDA graph's baked-in GPU pointers
        remain valid across substeps.  If the current color groups differ from the originals
        (i.e. dynamic recoloring ran last substep), the CUDA graph is invalidated so it will
        be rebuilt with the restored groups.
        """
        if self.model.particle_color_groups is not self._orig_particle_color_groups_gpu:
            if self.use_cuda_graph:
                self._particle_solve_graph = None
        self.model.particle_colors = self._orig_particle_colors_gpu
        self.model.particle_color_groups = self._orig_particle_color_groups_gpu

    def _apply_dynamic_recoloring(self, recolor_pairs_np: "np.ndarray") -> None:
        """Greedy CPU recoloring: reassign colors to separate same-color pairs in recolor region.

        For each pair (i, j) with the same color, reassign j (the higher-index vertex) to the
        smallest color not used by its mesh neighbors AND not the current color of i.
        This prevents both vertices from being updated in the same parallel sweep.

        Args:
            recolor_pairs_np: int32 numpy array of shape (2*K,) with pairs [i0, j0, i1, j1, ...].
        """
        import numpy as np

        colors_np = self._orig_particle_colors_np.copy()
        n_pairs = len(recolor_pairs_np) // 2
        n_applied = 0

        for k in range(n_pairs):
            i = int(recolor_pairs_np[2 * k])
            j = int(recolor_pairs_np[2 * k + 1])
            if colors_np[i] != colors_np[j]:
                # already separated by an earlier reassignment
                continue

            # Forbidden: all colors used by j's mesh neighbors, plus i's current color
            forbidden = {int(colors_np[nb]) for nb in self._recolor_neighbors[j]}
            forbidden.add(int(colors_np[i]))

            # Find smallest non-negative color not in forbidden
            new_color = 0
            while new_color in forbidden:
                new_color += 1

            colors_np[j] = new_color
            n_applied += 1

        # Rebuild color groups from updated colors array
        num_colors = int(colors_np.max()) + 1
        color_groups = [[] for _ in range(num_colors)]
        for v, c in enumerate(colors_np):
            color_groups[int(c)].append(v)

        # Push updated coloring back to GPU
        self.model.particle_colors = wp.array(colors_np, dtype=wp.int32, device=self.device)
        self.model.particle_color_groups = [
            wp.array(g, dtype=wp.int32, device=self.device)
            for g in color_groups
            if len(g) > 0
        ]
        # Invalidate CUDA graph: the color-group GPU arrays just changed, so the graph's
        # baked-in pointers are stale.  It will be rebuilt next solve_particle_iteration.
        if self.use_cuda_graph:
            self._particle_solve_graph = None

        print(f"[recolor] applied={n_applied} total_colors={num_colors}")

    def _apply_dynamic_recoloring_from_gpu(self) -> None:
        """Download recolor pairs from GPU and apply greedy dynamic recoloring on CPU.

        Assumes _restore_original_colors() has already been called before the watchlist
        kernel ran, so model.particle_colors currently holds the canonical static coloring.
        """
        rc = int(self.same_color_recolor_count.numpy()[0])
        if rc == 0:
            return

        # Cap at buffer capacity (excess pairs were dropped by kernel)
        rc_capped = min(rc, self.same_color_recolor_max_pairs)
        recolor_pairs_np = self.same_color_recolor_pairs.numpy()[: 2 * rc_capped]

        self._apply_dynamic_recoloring(recolor_pairs_np)

    # ------------------------------------------------------------------
    # Stage 5: Same-color watchlist barrier
    # ------------------------------------------------------------------

    def _apply_locked_vertex_recovery(self, state_out: State) -> None:
        """Apply escape gradient to locked vertices (conservative_bound < epsilon) after finalize.

        Called once per substep after finalize_particles. Walks each locked vertex's
        colliding triangles, accumulates repulsive escape gradient for near-contact pairs
        (dis < particle_self_contact_radius), and applies alpha * gradient directly to
        particle_q without OGC clamping.

        Logs per-substep metrics to CSV if recovery_log_path was set.
        Per-vertex diagnostics (d_before, d_after, dot_check) are written for the
        first 100 substeps to a separate CSV.
        """
        import csv
        import os

        self._recovery_locked_count.zero_()
        self._recovery_applied_count.zero_()
        self._recovery_sum_pen.zero_()
        self._recovery_max_pen.zero_()
        self._recovery_grad_dir_ok.zero_()
        self._recovery_grad_dir_total.zero_()
        self._recovery_new_pen_count.zero_()

        wp.launch(
            kernel=apply_locked_vertex_recovery_kernel,
            dim=self.model.particle_count,
            inputs=[
                state_out.particle_q,
                self.model.particle_flags,
                self.model.particle_inv_mass,
                self.particle_conservative_bounds,
                self.model.tri_indices,
                self.trimesh_collision_detector.collision_info,
                self.recovery_alpha,
                self.recovery_epsilon,
                self.particle_self_contact_radius,
                self.recovery_stiffness,
                self.model.particle_count,
            ],
            outputs=[
                self._recovery_applied_count,
                self._recovery_locked_count,
                self._recovery_sum_pen,
                self._recovery_max_pen,
                self._recovery_positions_buf,
                self._recovery_grad_dir_ok,
                self._recovery_grad_dir_total,
                self._recovery_new_pen_count,
                self._recovery_vertex_ids,
                self._recovery_d_before,
                self._recovery_d_after,
                self._recovery_dot_check,
            ],
            device=self.device,
        )

        locked = int(self._recovery_locked_count.numpy()[0])
        applied = int(self._recovery_applied_count.numpy()[0])
        sum_pen = float(self._recovery_sum_pen.numpy()[0])
        max_pen = float(self._recovery_max_pen.numpy()[0])
        avg_pen = (sum_pen / applied) if applied > 0 else 0.0
        grad_ok = int(self._recovery_grad_dir_ok.numpy()[0])
        grad_total = int(self._recovery_grad_dir_total.numpy()[0])
        new_pen = int(self._recovery_new_pen_count.numpy()[0])
        grad_ratio = (grad_ok / grad_total) if grad_total > 0 else 0.0
        tight_pen = int(self._tight_pen_count.numpy()[0])

        self._recovery_substep += 1

        # Console output for first 5 substeps
        if self._recovery_substep <= 5:
            print(
                f"[Recovery substep {self._recovery_substep:3d}] "
                f"locked={locked:4d}  applied={applied:4d}  "
                f"new_pen={new_pen:4d}  grad_dir_ratio={grad_ratio:.3f}  "
                f"tight_pen={tight_pen:4d}"
            )

        if self._recovery_log_path:
            # Summary CSV
            csv_path = os.path.join(
                self._recovery_log_path,
                f"recovery_metrics_{self.recovery_alpha:.4f}.csv",
            )
            write_header = not os.path.exists(csv_path)
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(
                        ["substep", "locked_count", "recovery_applied",
                         "avg_pen_depth", "max_pen_depth",
                         "new_pen_count", "grad_dir_ratio", "tight_pen_count"]
                    )
                writer.writerow(
                    [self._recovery_substep, locked, applied,
                     f"{avg_pen:.6f}", f"{max_pen:.6f}",
                     new_pen, f"{grad_ratio:.4f}", tight_pen]
                )

            # Per-vertex diagnostic CSV for first 100 substeps
            if self._recovery_substep <= 100 and applied > 0:
                n = min(applied, self.model.particle_count)
                vids = self._recovery_vertex_ids.numpy()[:n]
                d_bef = self._recovery_d_before.numpy()[:n]
                d_aft = self._recovery_d_after.numpy()[:n]
                dot_c = self._recovery_dot_check.numpy()[:n]

                vdiag_path = os.path.join(
                    self._recovery_log_path,
                    f"recovery_vertex_diag_{self.recovery_alpha:.4f}.csv",
                )
                vwrite_header = not os.path.exists(vdiag_path)
                with open(vdiag_path, "a", newline="") as vf:
                    vwriter = csv.writer(vf)
                    if vwrite_header:
                        vwriter.writerow(
                            ["substep", "vertex_id", "d_before", "d_after", "dot_grad_normal"]
                        )
                    for k in range(n):
                        vwriter.writerow(
                            [self._recovery_substep, int(vids[k]),
                             f"{d_bef[k]:.6f}", f"{d_aft[k]:.6f}", f"{dot_c[k]:.4f}"]
                        )

    def _launch_same_color_watchlist_barrier(self, current_state: State) -> None:
        """Launch accumulate_same_color_watchlist_barrier_kernel over all watchlist pairs.

        Adds IPC-like repulsive forces and hessians to particle_forces / particle_hessians
        for watchlist pairs with d < same_color_barrier_d_hat.

        Forces are scaled by inverse-mass weights (alpha_i = inv_mass_i/(inv_mass_i+inv_mass_j))
        to mitigate the double-push effect from simultaneous same-color updates.

        Must be called AFTER particle_forces.zero_() / particle_hessians.zero_() and BEFORE
        the color group solve loop so that the barrier forces are included in each vertex's solve.
        """
        wp.launch(
            kernel=accumulate_same_color_watchlist_barrier_kernel,
            dim=self.same_color_watchlist_max_pairs,
            inputs=[
                current_state.particle_q,
                self.model.particle_inv_mass,
                self.model.particle_flags,
                self.same_color_watchlist_pairs,
                self.same_color_watchlist_count,
                self.same_color_barrier_d_hat,
                self.same_color_barrier_stiffness,
            ],
            outputs=[
                self.particle_forces,
                self.particle_hessians,
            ],
            device=self.device,
        )
