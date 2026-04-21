"""
Cloth Twist Convergence Analysis
=================================
이 스크립트는 cloth_twist_reverse 예제를 headless로 실행하여
- 에너지 수렴 곡선 (kinetic energy, force residual norm)
- iteration당 residual 감소량

데이터를 수집하고 그래프를 저장합니다.

Usage:
    uv run python -m newton.examples.cloth.analyze_cloth_twist_convergence
    uv run python -m newton.examples.cloth.analyze_cloth_twist_convergence --num-frames 100
"""

import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")  # headless 환경용

import matplotlib.pyplot as plt
import numpy as np
import warp as wp
import warp.examples
from pxr import Usd

import newton
import newton.data_collector as dc
import newton.examples
import newton.usd
import newton.solvers
from newton import ParticleFlags
from newton.examples.cloth.example_cloth_twist_reverse import (
    apply_rotation,
    initialize_rotation,
)


# ---------------------------------------------------------------------------
# Headless viewer stub
# ---------------------------------------------------------------------------

class _HeadlessViewer:
    """No-op viewer that satisfies the Example interface."""

    def set_model(self, model): pass
    def set_camera(self, *args, **kwargs): pass
    def apply_forces(self, state): pass
    def begin_frame(self, t): pass
    def log_state(self, state): pass
    def end_frame(self): pass


# ---------------------------------------------------------------------------
# Instrumented Example
# ---------------------------------------------------------------------------

class AnalysisExample:
    """cloth_twist_reverse를 그대로 재현하되 data_collector + KE 추적을 추가."""

    def __init__(self, num_frames: int):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.iterations = 4
        self.bvh_rebuild_frames = 10
        self.rot_angular_velocity = math.pi / 3
        self.rot_end_time = 10
        self.num_frames = num_frames

        # data_collector 활성화 (per-iteration force_residual 기록용)
        dc.log_mode = dc.LOG_PERFORMANCE
        dc.record_frame_count = num_frames + 1  # 전 프레임 기록

        usd_stage = Usd.Stage.Open(
            os.path.join(warp.examples.get_asset_directory(), "square_cloth.usd")
        )
        usd_prim = usd_stage.GetPrimAtPath("/root/cloth/cloth")
        cloth_mesh = newton.usd.get_mesh(usd_prim)
        mesh_points = cloth_mesh.vertices
        mesh_indices = cloth_mesh.indices

        vertices = [wp.vec3(v) for v in mesh_points]
        self.faces = mesh_indices.reshape(-1, 3)

        scene = newton.ModelBuilder(gravity=0)
        scene.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(0, 0, 1), np.pi / 2),
            scale=0.01,
            vertices=vertices,
            indices=mesh_indices,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.2,
            tri_ke=1.0e3,
            tri_ka=1.0e3,
            tri_kd=2.0e-7,
            edge_ke=1e-3,
            edge_kd=1e-4,
        )
        scene.color()
        self.model = scene.finalize()
        self.model.soft_contact_ke = 1.0e3
        self.model.soft_contact_kd = 1.0e-4
        self.model.soft_contact_mu = 0.2

        cloth_size = 50
        left_side = [cloth_size - 1 + i * cloth_size for i in range(cloth_size)]
        right_side = [i * cloth_size for i in range(cloth_size)]
        rot_point_indices = left_side + right_side

        flags = self.model.particle_flags.numpy()
        for idx in rot_point_indices:
            flags[idx] = flags[idx] & ~ParticleFlags.ACTIVE
        self.model.particle_flags = wp.array(flags)

        self.solver = newton.solvers.SolverVBD(
            self.model,
            self.iterations,
            particle_enable_self_contact=True,
            particle_self_contact_radius=0.002,
            particle_self_contact_margin=0.0035,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.collision_pipeline = newton.examples.create_collision_pipeline(self.model)
        self.contacts = self.model.collide(
            self.state_0, collision_pipeline=self.collision_pipeline
        )

        rot_axes = [[0, 1, 0]] * len(right_side) + [[0, -1, 0]] * len(left_side)
        self.rot_point_indices = wp.array(rot_point_indices, dtype=int)
        self.t = wp.zeros((1,), dtype=float)
        self.rot_centers = wp.zeros(len(rot_point_indices), dtype=wp.vec3)
        self.rot_axes = wp.array(rot_axes, dtype=wp.vec3)
        self.roots = wp.zeros_like(self.rot_centers)
        self.roots_to_ps = wp.zeros_like(self.rot_centers)

        wp.launch(
            kernel=initialize_rotation,
            dim=self.rot_point_indices.shape[0],
            inputs=[
                self.rot_point_indices,
                self.state_0.particle_q,
                self.rot_centers,
                self.rot_axes,
                self.t,
            ],
            outputs=[self.roots, self.roots_to_ps],
        )

        # 에너지 & residual 기록 버퍼
        self.ke_history: list[float] = []          # frame별 kinetic energy
        self.residual_per_frame: list[float] = []  # frame별 마지막 iteration residual

    # ------------------------------------------------------------------
    # 내부 시뮬레이션 루프 (CUDA graph 없이 직접 실행)
    # ------------------------------------------------------------------

    def _compute_kinetic_energy(self) -> float:
        """현재 state_0에서 운동에너지를 계산합니다. (단위: J)"""
        vd = self.state_0.particle_qd.numpy()          # (N, 3)
        m = self.model.particle_mass.numpy()           # (N,)
        v2 = np.sum(vd ** 2, axis=1)                   # (N,)
        return float(0.5 * np.dot(m, v2))

    def simulate_frame(self, frame_idx: int):
        """한 프레임(= sim_substeps 번의 solver.step)을 실행합니다."""
        # data_collector 프레임 시작
        dc.frame_start(frame_idx)

        self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
        if frame_idx % self.bvh_rebuild_frames == 0:
            self.solver.rebuild_bvh(self.state_0)

        for substep in range(self.sim_substeps):
            # data_collector substep 시작
            dc.substep_start(substep)

            self.state_0.clear_forces()
            wp.launch(
                kernel=apply_rotation,
                dim=self.rot_point_indices.shape[0],
                inputs=[
                    self.rot_point_indices,
                    self.rot_axes,
                    self.roots,
                    self.roots_to_ps,
                    self.t,
                    self.rot_angular_velocity,
                    self.sim_dt,
                    self.rot_end_time,
                ],
                outputs=[
                    self.state_0.particle_q,
                    self.state_1.particle_q,
                ],
            )
            self.solver.step(
                self.state_0, self.state_1, self.control, self.contacts, self.sim_dt
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

        self.sim_time += self.frame_dt

        # 운동에너지 기록
        self.ke_history.append(self._compute_kinetic_energy())

    def run(self):
        print(f"Running {self.num_frames} frames (headless, no CUDA graph)...")
        for frame in range(self.num_frames):
            self.simulate_frame(frame)
            if frame % 10 == 0:
                print(f"  frame {frame:4d}/{self.num_frames}  KE={self.ke_history[-1]:.4e}")
        print("Done.")


# ---------------------------------------------------------------------------
# 데이터 추출 헬퍼
# ---------------------------------------------------------------------------

def extract_iteration_data():
    """data_collector.iteration_dict에서 NumPy 배열로 변환."""
    d = dc.iteration_dict
    if not d or not d.get("frame_idx"):
        return None
    return {k: np.array(v) for k, v in d.items()}


def select_representative_frames(num_frames: int, n: int = 6) -> list[int]:
    """구간을 균등 분할하여 대표 프레임 n개를 선택합니다."""
    return [int(round(i * (num_frames - 1) / (n - 1))) for i in range(n)]


# ---------------------------------------------------------------------------
# 플롯
# ---------------------------------------------------------------------------

def plot_all(example: AnalysisExample, output_dir: str = "."):
    idata = extract_iteration_data()
    frames = np.arange(len(example.ke_history))
    times = frames * example.frame_dt

    os.makedirs(output_dir, exist_ok=True)

    # ── 1. 운동에너지 수렴 곡선 ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(times, example.ke_history, color="steelblue", linewidth=1.5)
    ax.set_xlabel("Simulation time [s]")
    ax.set_ylabel("Kinetic energy [J]")
    ax.set_title("Kinetic Energy over Simulation Time\n(cloth twist → untwist)")
    ax.grid(alpha=0.3)

    # twist / untwist 경계 표시
    # angular_constant=5, angular_velocity=π/3  →  반환점: 5π / (π/3) = 15 s
    twist_peak_time = 5 * math.pi / (math.pi / 3)  # ≈ 15 s
    ax.axvline(x=twist_peak_time, color="tomato", linestyle="--",
               linewidth=1, label=f"Twist peak ≈ {twist_peak_time:.1f}s")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(output_dir, "kinetic_energy.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

    if idata is None:
        print("Warning: iteration_dict is empty — skipping residual plots.")
        return

    # ── 2. 대표 프레임의 iteration별 residual 감소 ──────────────────────
    rep_frames = select_representative_frames(example.num_frames)
    last_substep = example.sim_substeps - 1

    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.cm.viridis
    colors = [cmap(i / max(len(rep_frames) - 1, 1)) for i in range(len(rep_frames))]

    for color, fidx in zip(colors, rep_frames):
        mask = (
            (idata["frame_idx"] == fidx)
            & (idata["substep_idx"] == last_substep)
            & (idata["iteration_idx"] >= 0)   # -1(초기화 패스) 제외
        )
        if not np.any(mask):
            mask = (idata["frame_idx"] == fidx) & (idata["iteration_idx"] >= 0)
        if not np.any(mask):
            continue
        iters = idata["iteration_idx"][mask]
        residuals = idata["force_residual"][mask]
        order = np.argsort(iters)
        ax.semilogy(
            iters[order],
            residuals[order],
            marker="o",
            markersize=4,
            color=color,
            label=f"frame {fidx} (t={fidx * example.frame_dt:.1f}s)",
        )

    ax.set_xlabel("Iteration index")
    ax.set_ylabel("Force residual (mean |f| per particle)")
    ax.set_title("Force Residual per VBD Iteration\n(representative frames)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = os.path.join(output_dir, "residual_per_iteration.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

    # ── 3. 프레임별 최종 residual (수렴 추이) ───────────────────────────
    # 각 frame의 마지막 substep, 마지막 유효 iteration의 residual
    final_residuals = []
    for fidx in range(example.num_frames):
        mask = (
            (idata["frame_idx"] == fidx)
            & (idata["substep_idx"] == last_substep)
            & (idata["iteration_idx"] >= 0)
        )
        if not np.any(mask):
            mask = (idata["frame_idx"] == fidx) & (idata["iteration_idx"] >= 0)
        if not np.any(mask):
            final_residuals.append(np.nan)
            continue
        last_iter_val = idata["force_residual"][mask][
            np.argmax(idata["iteration_idx"][mask])
        ]
        final_residuals.append(float(last_iter_val))

    final_residuals = np.array(final_residuals)
    valid = ~np.isnan(final_residuals)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(times[valid], final_residuals[valid], color="darkorange", linewidth=1.5)
    ax.set_xlabel("Simulation time [s]")
    ax.set_ylabel("Force residual at last iteration")
    ax.set_title("Per-frame Final Force Residual over Time\n(convergence quality indicator)")
    ax.grid(alpha=0.3)
    ax.axvline(x=twist_peak_time, color="tomato", linestyle="--",
               linewidth=1, label=f"Twist peak ≈ {twist_peak_time:.1f}s")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(output_dir, "residual_convergence.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

    # ── 4. iteration당 residual 감소량 (line plot) ──────────────────────
    # X축: transition index  (0 = iter0→1, 1 = iter1→2, …)
    # Y축: r[i] - r[i+1]  (양수 = 감소)
    fig, ax = plt.subplots(figsize=(9, 5))
    for color, fidx in zip(colors, rep_frames):
        mask = (
            (idata["frame_idx"] == fidx)
            & (idata["substep_idx"] == last_substep)
            & (idata["iteration_idx"] >= 0)
        )
        if not np.any(mask):
            mask = (idata["frame_idx"] == fidx) & (idata["iteration_idx"] >= 0)
        if not np.any(mask):
            continue
        iters = idata["iteration_idx"][mask]
        residuals = idata["force_residual"][mask]
        order = np.argsort(iters)
        res_sorted = residuals[order]
        if len(res_sorted) < 2:
            continue
        drops = res_sorted[:-1] - res_sorted[1:]   # r[i] - r[i+1]
        transitions = np.arange(len(drops))         # 0, 1, 2, …
        ax.plot(
            transitions,
            drops,
            marker="o",
            markersize=5,
            color=color,
            label=f"frame {fidx} (t={fidx * example.frame_dt:.1f}s)",
        )

    ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
    ax.set_xlabel("Transition  (i → i+1)")
    ax.set_ylabel("Residual drop  r[i] − r[i+1]")
    ax.set_title("Residual Decrease per Iteration\n(positive = solver made progress)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(output_dir, "residual_drop_per_iteration.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

    # ── 5. CSV 내보내기 ──────────────────────────────────────────────────
    ke_csv = os.path.join(output_dir, "kinetic_energy.csv")
    np.savetxt(ke_csv, np.column_stack([frames, times, example.ke_history]),
               delimiter=",", header="frame,time_s,kinetic_energy_J", comments="")
    print(f"Saved: {ke_csv}")

    dc.set_record_name(os.path.join(output_dir, "cloth_twist"))
    dc.export_iteration_dict_to_csv("_iteration_data.csv")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cloth twist convergence analysis")
    parser.add_argument("--num-frames", type=int, default=300,
                        help="Number of frames to simulate (default: 300)")
    parser.add_argument("--output-dir", type=str, default="cloth_twist_analysis",
                        help="Directory to save plots and CSVs")
    args = parser.parse_args()

    example = AnalysisExample(num_frames=args.num_frames)
    example.run()
    plot_all(example, output_dir=args.output_dir)
    print(f"\nAll outputs saved to: {os.path.abspath(args.output_dir)}/")


if __name__ == "__main__":
    main()
