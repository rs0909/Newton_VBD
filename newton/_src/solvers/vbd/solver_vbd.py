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
    compute_jgs2_precomputation_numpy,
    _set_to_csr,
    accumulate_contact_force_and_hessian,
    accumulate_contact_force_and_hessian_log_collision,
    accumulate_contact_force_and_hessian_no_self_contact,
    accumulate_spring_force_and_hessian,
    build_edge_n_ring_edge_collision_filter,
    build_vertex_n_ring_tris_collision_filter,
    ANISOTROPIC_BOUND_MAX_HALF_SPACES,
    compute_particle_anisotropic_bound,
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
    # Augmented Lagrangian contact kernels
    update_al_multipliers_vt,
    update_al_multipliers_ee,
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
        coordinate_condensation: bool = False,
        use_al_contact: bool = False,
        al_mu: float = 0.0,
        al_Gamma: float = 0.9,
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

        """
        super().__init__(model)

        self.ogc_contact = ogc_contact
        self.use_coord_condensation = 1 if coordinate_condensation else 0
        if self.ogc_contact:
            print()
            print(">>> OGC Contact mode ON <<<")
            print()

        self.use_al_contact = use_al_contact
        if use_al_contact:
            if al_mu > 0.0:
                self.al_mu = al_mu
            elif model.tri_materials is not None and model.tri_materials.shape[0] > 0:
                # Section 5.2: μ = 0.1 × max_diag(K_elastic)
                # K_elastic diagonal ≈ max tri_ke (Young's modulus contribution per vertex)
                tri_ke_max = float(model.tri_materials.numpy()[:, 0].max())
                self.al_mu = 0.1 * tri_ke_max
            else:
                self.al_mu = float(model.soft_contact_ke)
            self.al_Gamma = al_Gamma
            print()
            print(f">>> AL Contact mode ON  (mu={self.al_mu:.3g}, Gamma={self.al_Gamma}) <<<")
            print()

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
            max_hs = int(ANISOTROPIC_BOUND_MAX_HALF_SPACES)
            self.particle_conservative_half_spaces = wp.zeros(
                (model.particle_count * max_hs,), dtype=wp.vec4, device=self.device
            )
            self.particle_conservative_n_half_spaces = wp.zeros(
                (model.particle_count,), dtype=wp.int32, device=self.device
            )
            self.particle_conservative_bounds = wp.zeros(
                (model.particle_count,), dtype=float, device=self.device
            )

            self.trimesh_collision_detector = TriMeshCollisionDetector(
                self.model,
                vertex_collision_buffer_pre_alloc=particle_vertex_contact_buffer_size,
                edge_collision_buffer_pre_alloc=particle_edge_contact_buffer_size,
                edge_edge_parallel_epsilon=particle_edge_parallel_epsilon,
                v_adj_edges=self.particle_adjacency.v_adj_edges,
                v_adj_edges_offsets=self.particle_adjacency.v_adj_edges_offsets,
                v_adj_faces=self.particle_adjacency.v_adj_faces,
                v_adj_faces_offsets=self.particle_adjacency.v_adj_faces_offsets,
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
        
        # JGS2 pre-computation: cubature weights  (Lan et al. 2025)
        if model.tri_count > 0 and self.particle_adjacency.v_adj_faces.size > 0:
            v_adj_faces_cpu    = self.particle_adjacency.v_adj_faces.to("cpu").numpy()
            v_adj_offsets_cpu  = self.particle_adjacency.v_adj_faces_offsets.to("cpu").numpy()
            cubature_np = compute_jgs2_precomputation_numpy(
                particle_count       = model.particle_count,
                pos_rest             = model.particle_q.numpy(),
                tri_indices          = model.tri_indices.numpy().reshape(-1, 3),
                tri_poses            = model.tri_poses.numpy(),
                tri_areas            = model.tri_areas.numpy(),
                tri_materials        = model.tri_materials.numpy(),
                v_adj_faces          = v_adj_faces_cpu,
                v_adj_faces_offsets  = v_adj_offsets_cpu,
            )
        else:
            cubature_np = np.zeros(0, dtype=np.float32)
        self.cubature_face_weights = wp.from_numpy(cubature_np, dtype=float, device=self.device)
        print(f"[JGS2] cubature_face_weights precomputed: {cubature_np.shape[0]} entries")

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
    
        self.initialize_rigid_bodies(state_in, contacts, dt, update_rigid_history)
        self.initialize_particles(state_in, dt)

        for iter_num in range(self.iterations):
            self.solve_rigid_body_iteration(state_in, state_out, contacts, dt)
            self.solve_particle_iteration(state_in, state_out, contacts, dt, iter_num, self.iterations-1)

            # AL multiplier update: λ, γ per contact pair after each VBD iteration
            if self.use_al_contact and self.particle_enable_self_contact:
                self._update_al_multipliers(state_out)

        self.finalize_rigid_bodies(state_out, dt)
        self.finalize_particles(state_out, dt)

    def initialize_particles(self, state_in: State, dt: float):
        """Initialize particle positions for the VBD iteration."""
        model = self.model

        # Early exit if no particles
        if model.particle_count == 0:
            return

        if self.particle_enable_self_contact:
            # Collision detection before initialization to compute conservative bounds
            if data_collector.is_log_nothing():
                self.collision_detection_penetration_free(state_in, -1)
            else:
                self.collision_detection_penetration_free_log_collision(state_in, -1)

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
                    self.particle_conservative_half_spaces,
                    self.particle_conservative_n_half_spaces,
                    self.particle_conservative_bounds,
                    self.inertia,
                ],
                dim=model.particle_count,
                device=self.device,
            )
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

        # Zero out forces and hessians
        self.particle_forces.zero_()
        self.particle_hessians.zero_()
        self.stvk_forces.zero_()
        if data_collector.is_log_collision():
            self.collision_counter.zero_()

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
                            self.particle_conservative_half_spaces,
                            self.particle_conservative_n_half_spaces,
                            self.particle_conservative_bounds,
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
                            self.particle_conservative_half_spaces,
                            self.particle_conservative_n_half_spaces,
                            self.particle_conservative_bounds,
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
                            self.cubature_face_weights,
                            self.use_coord_condensation,
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
                            self.cubature_face_weights,
                            self.use_coord_condensation,
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

        # Reset AL multipliers for the freshly detected contact set (lambda=0, gamma=1)
        if self.use_al_contact:
            self.trimesh_collision_detector.vt_al_lambda.fill_(0.0)
            self.trimesh_collision_detector.vt_al_gamma.fill_(1.0)
            self.trimesh_collision_detector.ee_al_lambda.fill_(0.0)
            self.trimesh_collision_detector.ee_al_gamma.fill_(1.0)

        self.pos_prev_collision_detection.assign(current_state.particle_q)
        wp.launch(
            kernel=compute_particle_conservative_bound,
            inputs=[
                self.particle_conservative_bound_relaxation,
                self.particle_self_contact_margin,
                self.particle_adjacency,
                self.trimesh_collision_detector.collision_info,
            ],
            outputs=[self.particle_conservative_bounds],
            dim=self.model.particle_count,
            device=self.device,
        )
        al_mu = self.al_mu if self.use_al_contact else 0.0
        wp.launch(
            kernel=compute_particle_anisotropic_bound,
            inputs=[
                self.particle_conservative_bound_relaxation,
                al_mu,
                self.pos_prev_collision_detection,
                self.model.tri_indices,
                self.model.edge_indices,
                self.particle_adjacency,
                self.trimesh_collision_detector.edge_edge_parallel_epsilon,
                self.trimesh_collision_detector.collision_info,
            ],
            outputs=[
                self.particle_conservative_half_spaces,
                self.particle_conservative_n_half_spaces,
            ],
            dim=self.model.particle_count,
            device=self.device,
        )


    def _update_al_multipliers(self, state_out: State):
        """Update AL multipliers (lambda, gamma) for all contact pairs after a VBD iteration.

        Algorithm 2, lines 22-27 from arXiv 2512.12151.
        Uses state_out.particle_q (positions after the solve) to evaluate constraint values.
        """
        model = self.model
        det = self.trimesh_collision_detector

        wp.launch(
            kernel=update_al_multipliers_vt,
            dim=model.particle_count,
            inputs=[
                state_out.particle_q,
                model.tri_indices,
                self.trimesh_collision_info,
                self.particle_self_contact_radius,
                self.al_mu,
                self.al_Gamma,
                det.vt_al_lambda,
                det.vt_al_gamma,
            ],
            device=self.device,
        )

        if model.edge_count > 0:
            wp.launch(
                kernel=update_al_multipliers_ee,
                dim=model.edge_count,
                inputs=[
                    state_out.particle_q,
                    model.edge_indices,
                    self.trimesh_collision_info,
                    self.particle_self_contact_radius,
                    self.al_mu,
                    self.al_Gamma,
                    self.trimesh_collision_detector.edge_edge_parallel_epsilon,
                    det.ee_al_lambda,
                    det.ee_al_gamma,
                ],
                device=self.device,
            )

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
            outputs=[self.particle_conservative_bounds],
            dim=self.model.particle_count,
            device=self.device,
        )
        al_mu = self.al_mu if self.use_al_contact else 0.0
        wp.launch(
            kernel=compute_particle_anisotropic_bound,
            inputs=[
                self.particle_conservative_bound_relaxation,
                al_mu,
                self.pos_prev_collision_detection,
                self.model.tri_indices,
                self.model.edge_indices,
                self.particle_adjacency,
                self.trimesh_collision_detector.edge_edge_parallel_epsilon,
                self.trimesh_collision_detector.collision_info,
            ],
            outputs=[
                self.particle_conservative_half_spaces,
                self.particle_conservative_n_half_spaces,
            ],
            dim=self.model.particle_count,
            device=self.device,
        )

    def al_debug_info(self):
        """Return a dict of AL contact diagnostic values (CPU readback; do not call in hot loops).

        Returns:
            dict with keys:
              al_mu           – penalty parameter μ
              iterations      – VBD iterations per substep
              n_vt_pairs      – total detected vertex-triangle contact slots
              n_ee_pairs      – total detected edge-edge contact slots
              max_vt_lambda   – max |λ| across all VT slots (proxy for penetration depth × μ)
              max_ee_lambda   – max |λ| across all EE slots
              hessian_norm_v0   – Frobenius norm of particle_hessians[0] (last substep, last iter)
              v0_vt_count       – detected VT contact count for vertex 0 (0 = no contacts detected)
              v0_active         – whether vertex 0 has ACTIVE particle flag
              max_hess_vertex   – index of vertex with highest hessian norm
              max_hess_norm     – that vertex's hessian Frobenius norm
              n_nonzero_hess    – number of vertices with non-zero hessian
              n_ee_slots_total  – total EE slots (counts each pair twice due to symmetric storage)
              n_ee_canonical    – unique EE pairs (slots/2), i.e. actual physical contact pairs
              ee_lambda_max_asymmetry – max |λ_e1→e2 - λ_e2→e1| across canonical pairs (should be 0)
              overflow_vt       – True if VT buffer overflowed (contacts dropped)
              overflow_ee       – True if EE buffer overflowed (contacts dropped)
              vt_overflow_count – number of vertices whose detected count > buffer size
              ee_overflow_count – number of edges whose detected count > buffer size
              vt_max_ratio      – max(count/buffer_size) across all vertices (>1.0 means overflow)
              ee_max_ratio      – max(count/buffer_size) across all edges
        """
        al_mu = getattr(self, "al_mu", float(self.model.soft_contact_ke))

        det = getattr(self, "trimesh_collision_detector", None)
        n_vt, n_ee, max_vt, max_ee = 0, 0, 0.0, 0.0
        v0_vt_count = 0
        n_ee_slots_total = 0
        n_ee_canonical = 0
        ee_lambda_max_asymmetry = 0.0
        overflow_vt = False
        overflow_ee = False
        vt_overflow_count = 0
        ee_overflow_count = 0
        vt_max_ratio = 0.0
        ee_max_ratio = 0.0
        if det is not None:
            if hasattr(det, "vertex_colliding_triangles_count"):
                vt_counts = det.vertex_colliding_triangles_count.numpy()
                n_vt = int(vt_counts.sum())
                v0_vt_count = int(vt_counts[0]) if len(vt_counts) > 0 else 0
            if hasattr(det, "edge_colliding_edges_count"):
                ee_counts = det.edge_colliding_edges_count.numpy()
                n_ee_slots_total = int(ee_counts.sum())
                n_ee = n_ee_slots_total  # keep backward compat name
            if hasattr(det, "vt_al_lambda"):
                vt_lam = det.vt_al_lambda.numpy()
                max_vt = float(np.max(np.abs(vt_lam))) if len(vt_lam) > 0 else 0.0
            if hasattr(det, "ee_al_lambda") and hasattr(det, "edge_colliding_edges_count"):
                ee_lam = det.ee_al_lambda.numpy()
                max_ee = float(np.max(np.abs(ee_lam))) if len(ee_lam) > 0 else 0.0

                # Check lambda asymmetry between e1→e2 and e2→e1 slots.
                # Symmetric storage means each canonical pair (e1<e2) has two lambda slots.
                # If updates are consistent, both should be equal at all times.
                if (hasattr(det, "edge_colliding_edges") and hasattr(det, "edge_colliding_edges_offsets")
                        and hasattr(self.model, "edge_count")):
                    ee_buf = det.edge_colliding_edges.numpy()         # int32, (2*n_slots,): [edge_idx, other_idx, ...]
                    ee_offsets = det.edge_colliding_edges_offsets.numpy()
                    ee_counts_arr = det.edge_colliding_edges_count.numpy()
                    n_edges = int(self.model.edge_count)
                    # Build map: (e1, e2) -> lambda for e1 < e2 canonical pair
                    lam_map = {}   # (min_e, max_e) -> list of lambda values
                    for e1 in range(n_edges):
                        cnt = int(ee_counts_arr[e1])
                        off = int(ee_offsets[e1])
                        for k in range(cnt):
                            e2 = int(ee_buf[2 * (off + k) + 1])
                            if e2 < 0:
                                continue
                            key = (min(e1, e2), max(e1, e2))
                            lam_val = float(ee_lam[off + k])
                            if key not in lam_map:
                                lam_map[key] = []
                            lam_map[key].append(lam_val)
                    n_ee_canonical = len(lam_map)
                    asym_vals = [abs(v[0] - v[1]) for v in lam_map.values() if len(v) == 2]
                    ee_lambda_max_asymmetry = float(max(asym_vals)) if asym_vals else 0.0

            # --- Buffer overflow check ---
            # resize_flags: [VT_overflow, Tri_overflow, EE_overflow, TriTri_overflow]
            if hasattr(det, "resize_flags"):
                rf = det.resize_flags.numpy()
                overflow_vt = bool(rf[0])
                overflow_ee = bool(rf[2])

            if hasattr(det, "vertex_colliding_triangles_count") and hasattr(det, "vertex_colliding_triangles_buffer_sizes"):
                vt_cnt = det.vertex_colliding_triangles_count.numpy()
                vt_buf = det.vertex_colliding_triangles_buffer_sizes.numpy()
                vt_overflow_mask = vt_cnt > vt_buf
                vt_overflow_count = int(vt_overflow_mask.sum())
                ratio = vt_cnt.astype(float) / np.maximum(vt_buf.astype(float), 1.0)
                vt_max_ratio = float(ratio.max()) if len(ratio) > 0 else 0.0

            if hasattr(det, "edge_colliding_edges_count") and hasattr(det, "edge_colliding_edges_buffer_sizes"):
                ee_cnt = det.edge_colliding_edges_count.numpy()
                ee_buf_sizes = det.edge_colliding_edges_buffer_sizes.numpy()
                ee_overflow_mask = ee_cnt > ee_buf_sizes
                ee_overflow_count = int(ee_overflow_mask.sum())
                ratio = ee_cnt.astype(float) / np.maximum(ee_buf_sizes.astype(float), 1.0)
                ee_max_ratio = float(ratio.max()) if len(ratio) > 0 else 0.0

        hess_norm = 0.0
        max_hess_vertex = -1
        max_hess_norm = 0.0
        n_nonzero_hess = 0
        v0_active = False
        if hasattr(self, "particle_hessians") and self.particle_hessians is not None:
            H = self.particle_hessians.numpy()
            h0 = H[0]
            hess_norm = float(np.linalg.norm(h0))
            norms = np.linalg.norm(H.reshape(H.shape[0], -1), axis=1)
            n_nonzero_hess = int((norms > 0).sum())
            if n_nonzero_hess > 0:
                max_hess_vertex = int(np.argmax(norms))
                max_hess_norm = float(norms[max_hess_vertex])

        if hasattr(self.model, "particle_flags") and self.model.particle_flags is not None:
            from newton import ParticleFlags
            flags = self.model.particle_flags.numpy()
            v0_active = bool(flags[0] & ParticleFlags.ACTIVE) if len(flags) > 0 else False

        return {
            "al_mu": al_mu,
            "iterations": self.iterations,
            "n_vt_pairs": n_vt,
            "n_ee_pairs": n_ee,
            "max_vt_lambda": max_vt,
            "max_ee_lambda": max_ee,
            "hessian_norm_v0": hess_norm,
            "v0_vt_count": v0_vt_count,
            "v0_active": v0_active,
            "max_hess_vertex": max_hess_vertex,
            "max_hess_norm": max_hess_norm,
            "n_nonzero_hess": n_nonzero_hess,
            "n_ee_slots_total": n_ee_slots_total,
            "n_ee_canonical": n_ee_canonical,
            "ee_lambda_max_asymmetry": ee_lambda_max_asymmetry,
            "overflow_vt": overflow_vt,
            "overflow_ee": overflow_ee,
            "vt_overflow_count": vt_overflow_count,
            "ee_overflow_count": ee_overflow_count,
            "vt_max_ratio": vt_max_ratio,
            "ee_max_ratio": ee_max_ratio,
        }

    def rebuild_bvh(self, state: State):
        """This function will rebuild the BVHs used for detecting self-contacts using the input `state`.

        When the simulated object deforms significantly, simply refitting the BVH can lead to deterioration of the BVH's
        quality. In these cases, rebuilding the entire tree is necessary to achieve better querying efficiency.

        Args:
            state (newton.State):  The state whose particle positions (:attr:`State.particle_q`) will be used for rebuilding the BVHs.
        """
        if self.particle_enable_self_contact:
            self.trimesh_collision_detector.rebuild(state.particle_q)
