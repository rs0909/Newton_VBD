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

"""VBD artificial-damping diagnostics framework.

Attach a :class:`VBDDiagnostics` object to :class:`SolverVBD` via the
``diagnostics`` constructor argument.  The solver calls hooks at key
pipeline stages; each hook syncs the GPU, reads the relevant arrays into
numpy, and accumulates records.  Call :meth:`VBDDiagnostics.save` (or use
it as a context manager) to flush data to a compressed .npz file.

All computation is CPU-side after ``wp.synchronize()``.  Diagnostics are
disabled when the ``diagnostics`` argument is ``None`` (the default), so
there is zero overhead in normal simulation runs.

Pipeline hooks (in order):
  before_prediction   — state before forward integration
  after_prediction    — state after inertia is computed (before solver)
  [before_truncation] — state of inertia array before Planar-DAT clip
  [after_truncation]  — state after Planar-DAT clip; truncation analysis
  after_solver        — state after all VBD iterations, before vel update
  end_of_step         — final state after velocity reconstruction

Optional per-iteration hook (log_iterations=True, disables CUDA graph):
  after_iter          — kinetic energy + residual after each iteration
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import numpy as np
import warp as wp

from newton._src.geometry.flags import ParticleFlags

if TYPE_CHECKING:
    from newton._src.sim import Model, State
    from newton._src.solvers.vbd.solver_vbd import SolverVBD


class VBDDiagnostics:
    """Records energy, momentum, and truncation metrics at every solver hook.

    Args:
        output_path: Path for the output .npz file (directory is created
            automatically).
        log_iterations: If True, also records kinetic energy and residual
            after every VBD iteration.  Forces CUDA-graph to be disabled for
            the entire run.
        log_frequency: Record only every ``log_frequency``-th substep (1 =
            every substep).
        log_truncation: If True, compute detailed before/after truncation
            analysis including normal/tangential decomposition.
    """

    def __init__(
        self,
        output_path: str,
        log_iterations: bool = False,
        log_frequency: int = 1,
        log_truncation: bool = True,
    ):
        self.output_path = output_path
        self.log_iterations = log_iterations
        self.log_frequency = log_frequency
        self.log_truncation = log_truncation

        self._step_num: int = 0
        self._substep_num: int = 0
        self._current_dt: float = 0.0

        # Accumulated records (list of dicts)
        self._step_records: list[dict] = []
        self._iter_records: list[dict] = []
        self._trunc_records: list[dict] = []

        # Temporary state saved in on_before_truncation, consumed in on_after_truncation
        self._pending_inertia_np: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Lifecycle helpers (called by user's simulation loop)
    # ------------------------------------------------------------------

    def begin_step(self, step_num: int, substep_num: int, dt: float) -> None:
        """Update the step/substep counters.  Call once before each solver.step()."""
        self._step_num = step_num
        self._substep_num = substep_num
        self._current_dt = dt

    def save(self) -> None:
        """Write accumulated data to ``output_path`` as a compressed NumPy archive."""
        out = self.output_path
        if not out.endswith(".npz"):
            out = out + ".npz"
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

        arrays: dict[str, np.ndarray] = {}

        # Step-level records: one entry per (tag, step)
        if self._step_records:
            for tag in sorted({r["tag"] for r in self._step_records}):
                recs = [r for r in self._step_records if r["tag"] == tag]
                prefix = f"{tag}/"
                arrays[prefix + "step"] = np.array([r["step"] for r in recs], dtype=np.int32)
                arrays[prefix + "substep"] = np.array([r["substep"] for r in recs], dtype=np.int32)
                arrays[prefix + "dt"] = np.array([r["dt"] for r in recs], dtype=np.float32)
                for key in recs[0]:
                    if key in ("tag", "step", "substep", "dt"):
                        continue
                    vals = [r.get(key, np.nan) for r in recs]
                    try:
                        arrays[prefix + key] = np.array(vals, dtype=np.float32)
                    except (TypeError, ValueError):
                        pass

        # Truncation records
        if self._trunc_records:
            prefix = "truncation/"
            arrays[prefix + "step"] = np.array([r["step"] for r in self._trunc_records], dtype=np.int32)
            arrays[prefix + "substep"] = np.array([r["substep"] for r in self._trunc_records], dtype=np.int32)
            for key in self._trunc_records[0]:
                if key in ("step", "substep"):
                    continue
                vals = [r.get(key, np.nan) for r in self._trunc_records]
                try:
                    arrays[prefix + key] = np.array(vals, dtype=np.float32)
                except (TypeError, ValueError):
                    pass

        # Iteration records
        if self._iter_records:
            prefix = "iterations/"
            for key in self._iter_records[0]:
                vals = [r.get(key, np.nan) for r in self._iter_records]
                try:
                    arrays[prefix + key] = np.array(vals, dtype=np.float32)
                except (TypeError, ValueError):
                    pass

        np.savez_compressed(out, **arrays)
        print(
            f"[VBDDiagnostics] saved {len(self._step_records)} step-records, "
            f"{len(self._trunc_records)} trunc-records, "
            f"{len(self._iter_records)} iter-records → {out}"
        )

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.save()

    # ------------------------------------------------------------------
    # Solver hooks (called by solver_vbd.py)
    # ------------------------------------------------------------------

    def on_before_prediction(self, state_in: "State", solver: "SolverVBD", dt: float) -> None:
        if not self._should_log():
            return
        wp.synchronize()
        q = state_in.particle_q.numpy().copy()
        qd = state_in.particle_qd.numpy().copy()
        m = self._step_metrics(q, qd, solver)
        m["n_contacts"] = float(self._count_contacts(solver))
        m["n_truncated"] = float(self._count_truncated(solver))
        self._step_records.append({"tag": "before_prediction", "step": self._step_num,
                                   "substep": self._substep_num, "dt": dt, **m})

    def on_after_prediction(self, state_in: "State", solver: "SolverVBD", dt: float) -> None:
        """After initialize_particles: state_in.particle_q holds the (possibly Planar-DAT-truncated) inertia."""
        if not self._should_log():
            return
        wp.synchronize()
        q = state_in.particle_q.numpy().copy()
        q_prev = solver.particle_q_prev.numpy().copy()
        qd = (q - q_prev) / max(dt, 1e-30)
        m = self._step_metrics(q, qd, solver)
        m["n_contacts"] = float(self._count_contacts(solver))
        m["n_truncated"] = float(self._count_truncated(solver))
        self._step_records.append({"tag": "after_prediction", "step": self._step_num,
                                   "substep": self._substep_num, "dt": dt, **m})

    def on_before_truncation(
        self,
        inertia: wp.array,
        state_in: "State",
        solver: "SolverVBD",
    ) -> None:
        """Called immediately before the Planar-DAT truncation in initialize_particles."""
        if not self._should_log() or not self.log_truncation:
            return
        wp.synchronize()
        self._pending_inertia_np = inertia.numpy().copy()

    def on_after_truncation(
        self,
        inertia: wp.array,
        state_in: "State",
        solver: "SolverVBD",
    ) -> None:
        """Called immediately after the Planar-DAT truncation in initialize_particles."""
        if not self._should_log() or not self.log_truncation:
            return
        if self._pending_inertia_np is None:
            return
        wp.synchronize()

        inertia_np = self._pending_inertia_np
        self._pending_inertia_np = None
        trunc_np = state_in.particle_q.numpy().copy()
        trunc_ts = solver.truncation_ts.numpy().copy()

        trunc_metrics = self._truncation_analysis(inertia_np, trunc_np, trunc_ts, state_in, solver)
        self._trunc_records.append({
            "step": self._step_num,
            "substep": self._substep_num,
            **trunc_metrics,
        })

    def on_after_solver(self, state_in: "State", solver: "SolverVBD", dt: float) -> None:
        """After the VBD iteration loop; velocities not yet reconstructed."""
        if not self._should_log():
            return
        wp.synchronize()
        q = state_in.particle_q.numpy().copy()
        q_prev = solver.particle_q_prev.numpy().copy()
        qd = (q - q_prev) / max(dt, 1e-30)
        m = self._step_metrics(q, qd, solver)
        m["n_contacts"] = float(self._count_contacts(solver))
        m["n_truncated"] = float(self._count_truncated(solver))
        m["residual_mean"], m["residual_max"] = self._force_residual(solver)
        self._step_records.append({"tag": "after_solver", "step": self._step_num,
                                   "substep": self._substep_num, "dt": dt, **m})

    def on_end_of_step(self, state_out: "State", solver: "SolverVBD", dt: float) -> None:
        """After finalize_particles: velocities reconstructed via BDF-1."""
        if not self._should_log():
            return
        wp.synchronize()
        q = state_out.particle_q.numpy().copy()
        qd = state_out.particle_qd.numpy().copy()
        m = self._step_metrics(q, qd, solver)
        m["n_contacts"] = float(self._count_contacts(solver))
        m["n_truncated"] = float(self._count_truncated(solver))
        self._step_records.append({"tag": "end_of_step", "step": self._step_num,
                                   "substep": self._substep_num, "dt": dt, **m})

    def on_after_solver_iteration(
        self,
        state_in: "State",
        solver: "SolverVBD",
        dt: float,
        iter_num: int,
    ) -> None:
        """Called after each VBD iteration (only when log_iterations=True)."""
        if not self._should_log():
            return
        wp.synchronize()
        q = state_in.particle_q.numpy().copy()
        q_prev = solver.particle_q_prev.numpy().copy()
        qd = (q - q_prev) / max(dt, 1e-30)
        mass = solver.model.particle_mass.numpy()
        v_sq = np.sum(qd ** 2, axis=1)
        ke = float(0.5 * np.dot(mass, v_sq))
        res_mean, res_max = self._force_residual(solver)
        self._iter_records.append({
            "step": float(self._step_num),
            "substep": float(self._substep_num),
            "iter": float(iter_num),
            "kinetic_energy": ke,
            "residual_mean": res_mean,
            "residual_max": res_max,
        })

    # ------------------------------------------------------------------
    # Internal: metrics computation
    # ------------------------------------------------------------------

    def _should_log(self) -> bool:
        return self._substep_num % self.log_frequency == 0

    def _step_metrics(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        solver: "SolverVBD",
    ) -> dict:
        """Compute all energy and momentum quantities from numpy arrays."""
        model = solver.model
        mass = model.particle_mass.numpy()          # (N,)
        flags = model.particle_flags.numpy()        # (N,) uint32
        gravity_np = model.gravity.numpy()[0]       # (3,) via wp.vec3 → shape (1,3)

        active = (flags & int(ParticleFlags.ACTIVE)) != 0
        q_a = q[active]       # (Na, 3)
        qd_a = qd[active]     # (Na, 3)
        m_a = mass[active]    # (Na,)
        total_mass = float(np.sum(m_a))

        # Kinetic energy
        v_sq = np.sum(qd_a ** 2, axis=1)
        ke = float(0.5 * np.dot(m_a, v_sq))

        # Linear momentum
        lin_mom = (m_a[:, np.newaxis] * qd_a).sum(axis=0)  # (3,)

        # Centre of mass
        com = (m_a[:, np.newaxis] * q_a).sum(axis=0) / max(total_mass, 1e-30)
        com_vel = lin_mom / max(total_mass, 1e-30)

        # Angular momentum about COM
        r = q_a - com
        mv = m_a[:, np.newaxis] * qd_a
        ang_mom = np.cross(r, mv).sum(axis=0)  # (3,)

        # Gravitational potential (E = m * g_mag * height)
        g_mag = float(np.linalg.norm(gravity_np))
        if g_mag > 1e-12:
            g_hat = gravity_np / g_mag
            heights = -(q_a @ g_hat)
            gpe = float(np.dot(m_a, g_mag * heights))
        else:
            gpe = 0.0

        stretch_e = _compute_stretch_energy(q, model)
        bending_e = _compute_bending_energy(q, model)
        total_e = ke + stretch_e + bending_e + gpe

        return {
            "kinetic_energy": ke,
            "stretch_energy": stretch_e,
            "bending_energy": bending_e,
            "gravitational_potential": gpe,
            "total_energy": total_e,
            "linear_momentum_x": float(lin_mom[0]),
            "linear_momentum_y": float(lin_mom[1]),
            "linear_momentum_z": float(lin_mom[2]),
            "angular_momentum_x": float(ang_mom[0]),
            "angular_momentum_y": float(ang_mom[1]),
            "angular_momentum_z": float(ang_mom[2]),
            "com_x": float(com[0]),
            "com_y": float(com[1]),
            "com_z": float(com[2]),
            "com_vel_x": float(com_vel[0]),
            "com_vel_y": float(com_vel[1]),
            "com_vel_z": float(com_vel[2]),
        }

    def _force_residual(self, solver: "SolverVBD") -> tuple[float, float]:
        """Return (mean, max) of ||F_total|| across all particles."""
        try:
            total_f = (solver.particle_forces + solver.stvk_forces).numpy().reshape(-1, 3)
            mag = np.linalg.norm(total_f, axis=1)
            return float(np.mean(mag)), float(np.max(mag))
        except Exception:
            return float("nan"), float("nan")

    def _count_contacts(self, solver: "SolverVBD") -> int:
        if not solver.particle_enable_self_contact:
            return 0
        try:
            det = solver.trimesh_collision_detector
            info = det.collision_info
            vt = int(np.sum(np.minimum(
                info.vertex_colliding_triangles_count.numpy(),
                info.vertex_colliding_triangles_buffer_sizes.numpy(),
            )))
            ee = int(np.sum(np.minimum(
                info.edge_colliding_edges_count.numpy(),
                info.edge_colliding_edges_buffer_sizes.numpy(),
            )))
            return vt + ee
        except Exception:
            return -1

    def _count_truncated(self, solver: "SolverVBD") -> int:
        if not solver.particle_enable_self_contact or not solver.use_planar_dat:
            return 0
        try:
            ts = solver.truncation_ts.numpy()
            return int(np.sum(ts < 0.999))
        except Exception:
            return -1

    def _truncation_analysis(
        self,
        inertia_np: np.ndarray,
        trunc_np: np.ndarray,
        trunc_ts: np.ndarray,
        state_in: "State",
        solver: "SolverVBD",
    ) -> dict:
        """Before/after Planar-DAT truncation analysis.

        Computes delta-energy, normal/tangential removed displacement, and
        per-vertex truncation scale.  Contact normals come from the
        vertex_colliding_triangles buffer (first listed collision per vertex).
        """
        n_v = len(inertia_np)
        model = solver.model
        mass = model.particle_mass.numpy()  # (N,)
        flags = model.particle_flags.numpy()
        gravity_np = model.gravity.numpy()[0]
        active = (flags & int(ParticleFlags.ACTIVE)) != 0

        # Removed displacement: inertia → truncated
        removed = inertia_np - trunc_np  # (N, 3)
        removed_mag = np.linalg.norm(removed, axis=1)

        # Truncation characterisation
        n_truncated = int(np.sum(trunc_ts < 0.999))
        fraction_truncated = n_truncated / max(n_v, 1)
        total_removed = float(np.sum(removed_mag[active]))

        # Delta kinetic energy
        q_prev = solver.pos_prev_collision_detection.numpy().copy()
        dt = self._current_dt
        qd_before = (inertia_np - q_prev) / max(dt, 1e-30)
        qd_after = (trunc_np - q_prev) / max(dt, 1e-30)
        v_sq_before = np.sum(qd_before ** 2, axis=1)
        v_sq_after = np.sum(qd_after ** 2, axis=1)
        dke = float(0.5 * np.dot(mass[active], (v_sq_after - v_sq_before)[active]))

        # Delta stretch energy
        dstretch = (_compute_stretch_energy(trunc_np, model)
                    - _compute_stretch_energy(inertia_np, model))

        # Delta bending energy
        dbending = (_compute_bending_energy(trunc_np, model)
                    - _compute_bending_energy(inertia_np, model))

        # Delta gravitational potential
        g_mag = float(np.linalg.norm(gravity_np))
        g_hat = gravity_np / max(g_mag, 1e-12)
        dheights = -((trunc_np[active] - inertia_np[active]) @ g_hat)
        dgpe = float(np.dot(mass[active], g_mag * dheights))

        dtotal = dke + dstretch + dbending + dgpe

        # Delta linear momentum
        dp = ((mass[active, np.newaxis] * (qd_after - qd_before)[active])
              .sum(axis=0))

        # Normal / tangential decomposition of removed displacement
        r_normal_arr = np.zeros(n_v, dtype=np.float64)
        r_tangent_arr = np.zeros(n_v, dtype=np.float64)
        normals_used = 0

        if solver.particle_enable_self_contact:
            try:
                q_cur = state_in.particle_q.numpy()
                tri_indices = model.tri_indices.numpy()  # (T, 3)
                det = solver.trimesh_collision_detector
                info = det.collision_info

                vt_counts_np = np.minimum(
                    info.vertex_colliding_triangles_count.numpy(),
                    info.vertex_colliding_triangles_buffer_sizes.numpy(),
                )
                vt_flat = info.vertex_colliding_triangles.numpy()
                vt_offsets = info.vertex_colliding_triangles_offsets.numpy()

                for v in range(n_v):
                    if removed_mag[v] < 1e-10:
                        continue
                    n_cols = int(vt_counts_np[v])
                    if n_cols == 0:
                        continue

                    # Use the first listed collision triangle
                    offset = int(vt_offsets[v])
                    tri_idx = int(vt_flat[2 * offset + 1])
                    if tri_idx < 0 or tri_idx >= len(tri_indices):
                        continue

                    t0 = q_cur[tri_indices[tri_idx, 0]]
                    t1 = q_cur[tri_indices[tri_idx, 1]]
                    t2 = q_cur[tri_indices[tri_idx, 2]]
                    n_vec = np.cross(t1 - t0, t2 - t0)
                    n_len = np.linalg.norm(n_vec)
                    if n_len < 1e-12:
                        continue
                    n_hat = n_vec / n_len

                    # Orient toward vertex
                    if np.dot(q_cur[v] - t0, n_hat) < 0:
                        n_hat = -n_hat

                    r = removed[v]
                    r_n = float(np.dot(r, n_hat))
                    r_t = float(np.linalg.norm(r - r_n * n_hat))
                    r_normal_arr[v] = r_n
                    r_tangent_arr[v] = r_t
                    normals_used += 1
            except Exception:
                pass

        total_r_normal = float(np.sum(np.abs(r_normal_arr)))
        total_r_tangent = float(np.sum(r_tangent_arr))
        tangential_fraction = total_r_tangent / max(total_r_normal + total_r_tangent, 1e-30)

        return {
            "n_truncated": float(n_truncated),
            "fraction_truncated": fraction_truncated,
            "total_removed_displacement": total_removed,
            "trunc_scale_min": float(np.min(trunc_ts)),
            "trunc_scale_mean": float(np.mean(trunc_ts)),
            "delta_kinetic_energy": dke,
            "delta_stretch_energy": dstretch,
            "delta_bending_energy": dbending,
            "delta_gravitational_potential": dgpe,
            "delta_total_energy": dtotal,
            "delta_linear_momentum_x": float(dp[0]),
            "delta_linear_momentum_y": float(dp[1]),
            "delta_linear_momentum_z": float(dp[2]),
            "total_r_normal": total_r_normal,
            "total_r_tangent": total_r_tangent,
            "tangential_fraction": tangential_fraction,
            "contact_normals_used": float(normals_used),
        }


# ------------------------------------------------------------------
# Pure-numpy energy helpers (module-level, reused across instances)
# ------------------------------------------------------------------

def _compute_stretch_energy(q: np.ndarray, model: "Model") -> float:
    """StVK conservative stretch energy summed over all triangles.

    ψ = mu * ||G||_F^2 + 0.5 * lambda * trace(G)^2,
    where G = 0.5 * (F^T F - I) is the Green–Lagrange strain tensor.
    """
    if model.tri_count == 0:
        return 0.0
    q3 = q.reshape(-1, 3)
    tri_idx = model.tri_indices.numpy()       # (T, 3)
    tri_pose = model.tri_poses.numpy()        # (T, 2, 2)  — DmInv
    tri_area = model.tri_areas.numpy()        # (T,)
    tri_mat = model.tri_materials.numpy()     # (T, 5)  ke, ka, kd, drag, lift
    mu = tri_mat[:, 0]
    lmbd = tri_mat[:, 1]

    x0 = q3[tri_idx[:, 0]]
    x1 = q3[tri_idx[:, 1]]
    x2 = q3[tri_idx[:, 2]]
    x01 = x1 - x0
    x02 = x2 - x0

    # F = [x01, x02] * DmInv
    f0 = (x01 * tri_pose[:, 0, 0:1] + x02 * tri_pose[:, 1, 0:1])  # (T,3)
    f1 = (x01 * tri_pose[:, 0, 1:2] + x02 * tri_pose[:, 1, 1:2])  # (T,3)

    f0f0 = np.einsum("ti,ti->t", f0, f0)
    f1f1 = np.einsum("ti,ti->t", f1, f1)
    f0f1 = np.einsum("ti,ti->t", f0, f1)

    G00 = 0.5 * (f0f0 - 1.0)
    G11 = 0.5 * (f1f1 - 1.0)
    G01 = 0.5 * f0f1

    G_frob_sq = G00 ** 2 + G11 ** 2 + 2.0 * G01 ** 2
    trace_G = G00 + G11
    psi = mu * G_frob_sq + 0.5 * lmbd * trace_G ** 2
    return float(np.dot(psi, tri_area))


def _compute_bending_energy(q: np.ndarray, model: "Model") -> float:
    """Dihedral-angle bending energy summed over all interior edges.

    E = k * l_rest * (theta - theta_rest)^2
    """
    if model.edge_count == 0:
        return 0.0
    q3 = q.reshape(-1, 3)
    edge_idx = model.edge_indices.numpy()            # (E, 4): [opp0, opp1, e_start, e_end]
    edge_rest_angle = model.edge_rest_angle.numpy()  # (E,)
    edge_rest_length = model.edge_rest_length.numpy()# (E,)
    edge_bend = model.edge_bending_properties.numpy()# (E, 2): [ke, kd]
    ke = edge_bend[:, 0]

    # Interior edges only (boundary edges have opp vertex == -1)
    valid = (edge_idx[:, 0] >= 0) & (edge_idx[:, 1] >= 0)
    if not np.any(valid):
        return 0.0

    vi0 = edge_idx[valid, 0]
    vi1 = edge_idx[valid, 1]
    vi2 = edge_idx[valid, 2]
    vi3 = edge_idx[valid, 3]
    ke_v = ke[valid]
    l_v = edge_rest_length[valid]
    theta_rest_v = edge_rest_angle[valid]

    x0, x1, x2, x3 = q3[vi0], q3[vi1], q3[vi2], q3[vi3]
    x02 = x2 - x0
    x03 = x3 - x0
    x13 = x3 - x1
    x12 = x2 - x1
    e = x3 - x2

    n1 = np.cross(x02, x03)
    n2 = np.cross(x13, x12)
    e_vec = e

    n1_len = np.linalg.norm(n1, axis=1, keepdims=True)
    n2_len = np.linalg.norm(n2, axis=1, keepdims=True)
    e_len = np.linalg.norm(e_vec, axis=1, keepdims=True)
    degenerate = (n1_len[:, 0] < 1e-8) | (n2_len[:, 0] < 1e-8) | (e_len[:, 0] < 1e-8)

    n1_hat = n1 / (n1_len + 1e-30)
    n2_hat = n2 / (n2_len + 1e-30)
    e_hat = e_vec / (e_len + 1e-30)

    cos_t = np.einsum("ti,ti->t", n1_hat, n2_hat)
    cross_n = np.cross(n1_hat, n2_hat)
    sin_t = np.einsum("ti,ti->t", cross_n, e_hat)
    theta = np.arctan2(sin_t, cos_t)

    k_total = ke_v * l_v
    E = k_total * (theta - theta_rest_v) ** 2
    E[degenerate] = 0.0
    return float(np.sum(E))
