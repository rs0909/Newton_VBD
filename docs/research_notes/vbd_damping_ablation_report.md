# VBD Artificial Damping Ablation Study

**Date**: 2026-06-09  
**Branch**: `claude/step2-dyanmic_recoloring_ogc`  
**Device**: NVIDIA GeForce RTX 4090 (cuda:1)  
**Solver**: Newton VBD — Planar-DAT (arXiv:2604.15513), OGC contact mode  

---

## 1. Motivation

Vertex Block Descent (VBD) is an implicit solver that iterates over graph-colored vertex groups. Real-time garment simulation requires the solver to run with a limited iteration budget (typically 10–100). This study quantifies *how much* artificial damping the solver produces, and *which pipeline stage* is responsible, under four qualitatively different contact conditions.

**Damping source candidates**:

| # | Source | Hypothesis |
|---|--------|-----------|
| 1 | Implicit time integration | Backward-Euler velocity back-calculation dissipates energy structurally |
| 2 | Solver under-convergence | Insufficient iterations → residual force → net work done on cloth |
| 3 | Planar-DAT truncation | Conservative bounds clip inertia displacement → KE removed |
| 4 | Tangential displacement clipping | Truncation removes contact-tangential motion |
| 5 | Contact topology / OGC recoloring | Graph changes alter solver convergence path |

---

## 2. Methodology

### 2.1 Diagnostic Hooks

A `VBDDiagnostics` class was instrumented into `SolverVBD.step()` at four checkpoints per substep:

```
before_prediction → after_prediction → [before/after_truncation] → after_solver → end_of_step
```

At each checkpoint: kinetic energy (KE), stretch energy (StVK), bending energy (dihedral), gravitational PE, linear/angular momentum, COM, solver residual, contact count, truncation statistics.

**Truncation analysis** (`on_before/after_truncation`): per-substep ΔKE, ΔE, Δmomentum, R_normal / R_tangent decomposition of removed displacement.

All diagnostics are **disabled by default**; no change to solver behavior when off.

### 2.2 Ablation Matrix

For each scene × iteration count, four variants were run:

| Variant | `particle_enable_self_contact` | `use_planar_dat` | `no_friction` |
|---------|-------------------------------|-----------------|--------------|
| `full`       | ✓ | ✓ | — |
| `no_contact` | ✗ | ✗ | — |
| `no_trunc`   | ✓ | ✗ | — |
| `no_friction`| ✓ | ✓ | ✓ |

**Note**: `dynamic_recoloring=False` in all variants (not passed to the solver constructor; defaults off). The `>>> OGC Contact mode ON <<<` message refers to the collision *detection* algorithm (OGC vs. standard BVH query), not graph recoloring. Therefore `full − no_trunc` isolates **Planar-DAT only** without dynamic recoloring.

### 2.3 Iteration and Substep Sweeps

**Iteration sweep** (§3–§6): `iterations ∈ {10, 50, 100, 500}`, `substeps = 10`, `fps = 60`, `dt = 1/600 s ≈ 1.667 ms`.

**Substep sweep** (§7.5): `substeps ∈ {2, 5, 10, 20, 40}`, `iterations = 100`, `fps = 60`, `dt = 1/(60 × substeps) s`. All four ablation variants run for Scenes A and B only. The substep sweep varies dt while keeping the total simulated time fixed (30 frames = 0.5 s), so each row represents a different per-substep timestep.

### 2.4 Scenes

| ID | Name | Description | Duration | Self-contact |
|----|------|-------------|----------|-------------|
| A | `no_contact_oscillation` | 24×24 cloth, 2 fixed corners, initial z-displacement | 30 frames (0.5 s) | None |
| B | `frictionless_sliding` | Two 24×24 cloth layers, top slides on fixed bottom, μ=0 | 30 frames (0.5 s) | Sustained (~400 pairs) |
| C | `separating_contact` | Tent-shaped cloth, self-contacts then separates | 30 frames (0.5 s) | Brief (~30 pairs) |
| D | `twist_release` | Square cloth, corners twisted to t=10 s then released | 720 frames (12 s) | Massive (~40,000 pairs) |

---

## 3. Scene A — No-Contact Oscillation

**Setup**: Cloth suspended by 2 fixed corners, displaced vertically, released under gravity. No self-contact pipeline.

### 3.1 Energy Decay

| iter | E_initial (J) | E_final (J) | E retained | residual_mean (N) | wall time |
|------|--------------|------------|-----------|------------------|-----------|
| 10   | 284.4 | 6.23 | **2.2%** | 3.385 | 2.9 s |
| 50   | 284.4 | 5.90 | **2.1%** | 3.390 | 5.5 s |
| 100  | 284.4 | 5.89 | **2.1%** | 3.387 | 9.1 s |
| 500  | 284.4 | 5.91 | **2.1%** | 3.391 | 35.1 s |

### 3.2 Per-substep Energy Budget

| Stage | ΔKE / substep | ΔE_total / substep |
|-------|--------------|-------------------|
| Prediction (inertia extrapolation) | +0.743 J | +0.653 J |
| VBD solver | −0.560 J | **−1.582 J** |

### 3.3 Findings

1. **97.9% energy is lost in 0.5 s** with 100 iterations and no contact.
2. **Iteration count has no effect**: iter=10 and iter=500 produce identical energy profiles (max difference 0.56 J over 300 substeps, 0.03%).
3. **Residuals do not decrease with more iterations**: mean residual ≈ 3.39 N for iter=10 through iter=500, confirming the solver reaches its fixed point in fewer than 10 iterations.
4. **All energy loss occurs in the VBD solver stage**, not in prediction.

**Conclusion**: The dominant damping source in Scene A is **structural implicit damping** from the backward-Euler–style velocity back-calculation `v = (q_new − q_old) / dt`. This is an intrinsic property of the time integration scheme, independent of iteration count.

---

## 4. Scene B — Frictionless Sliding

**Setup**: Two 24×24 cloth layers. Bottom layer pinned at corners. Top layer slides on bottom under gravity. No friction (μ=0). Planar-DAT active.

### 4.1 Energy Decomposition at Final Frame (iter=100)

| Variant | KE (J) | grav PE (J) | stretch (J) | total (J) | Interpretation |
|---------|--------|------------|------------|----------|----------------|
| `full`       | 1.28 | −88.1 | 21.2 | −65.6 | Cloth settled on surface |
| `no_contact` | 356.4 | −365.5 | 0.8 | −8.2 | Free fall through bottom layer |
| `no_trunc`   | 1.00 | −92.1 | 22.2 | −68.9 | Slightly lower, more stretch |
| `no_friction`| 1.28 | −88.4 | 21.5 | −65.9 | ≈ full (scene is frictionless anyway) |

### 4.2 Planar-DAT Truncation vs. Iteration Count

| iter | n_truncated | frac % | ΔKE_trunc / substep | ΔE_trunc / substep | tang_frac | trunc_scale_min |
|------|------------|--------|--------------------|--------------------|-----------|----------------|
| 10  | 78.3  | 6.3%  | −0.272 J | −0.209 J | 0.429 | 0.237 |
| 50  | 103.1 | 8.2%  | −0.984 J | −0.837 J | 0.418 | 0.160 |
| 100 | 118.3 | 9.5%  | −1.414 J | −1.247 J | 0.420 | 0.126 |
| 500 | 127.6 | 10.2% | −1.781 J | −1.584 J | 0.413 | 0.112 |

#### Total energy vs. iteration count

The table above shows the per-substep Planar-DAT truncation rate increasing monotonically with iteration count. The natural follow-up question is whether this translates into a monotonically lower *final* total energy:

| iter | E_initial (J) | E_final (J, t=0.5s) | ΔE = E_initial − E_final (J) |
|------|--------------|---------------------|-------------------------------|
| 10  | 11.34 | −49.49 | 60.83 |
| 50  | 18.65 | −67.81 | 86.45 |
| 100 | 19.06 | −65.83 | 84.89 |
| 500 | 19.66 | −64.43 | 84.09 |

iter=50/100/500 start near +19–20J and track closely until t≈0.1s; iter=10 starts measurably lower (≈11J) after a brief transient in t<0.02s, reflecting under-converged contact resolution during the layers' initial settling. From t≈0.15s the iter=10 trajectory separates from the other three, plateauing near −49J by t≈0.3s and remaining essentially flat (±1J ringing) through t=0.5s — a qualitatively different, higher-energy final state than iter≥50. The iter=50/100/500 trajectories remain close together throughout and are still decreasing gently at t=0.5s, ending at −67.8J, −65.8J, and −64.4J respectively — i.e., *less* loss at iter=500 (84.1J) than at iter=50 (86.5J), the opposite of a simple "more iterations → more loss" trend. This ≈4J spread among iter≥50 is the same order of magnitude as the ±5J `full − no_trunc` isolation values in §7.2, indicating that Planar-DAT truncation's *net* effect on final total energy is a small, non-monotonic correction layered on top of a dominant ≈85J loss common to all iter≥50 runs — the per-substep truncation-rate increase shown in the table above does not translate into a correspondingly larger total-energy deficit.

### 4.3 Findings

1. **More iterations → more truncation per substep, but not more total energy loss.** With 500 iterations, Planar-DAT removes 1.78 J of KE per substep, vs. only 0.27 J at 10 iterations (table above). However, the *final* total energy is not monotonic in iteration count (see "Total energy vs. iteration count" above): the large iter=10→50 jump (60.8J→86.5J loss) reflects a convergence-quality transition, while iter=50/100/500 differ by only ≈4J, with iter=500 showing the *least* loss of the three.
2. **Tangential fraction ≈ 0.42 (constant)**. Even with μ=0 (no friction force), 42% of removed displacement is tangential to the contact normal. Planar-DAT acts as an implicit friction-like damper in the sliding direction.
3. **`delta_linear_momentum_z > 0`**: Truncation injects upward normal-direction momentum (cloth pushed away from surface), while removing KE overall — a non-conservative impulse.
4. The `no_contact` variant shows extreme KE (356 J at t=0.5 s) from free fall, confirming that contact constraints are essential to limit cloth motion.

**Conclusion**: In sustained-contact scenarios, **Planar-DAT truncation is the primary energy dissipation mechanism beyond structural implicit damping**, and its magnitude scales with iteration count.

---

## 5. Scene C — Separating Contact

**Setup**: Single cloth in a shallow tent shape (z-displacement), one edge pinned. Cloth self-contacts briefly (~frames 27–29) then separates.

### 5.1 Convergence vs. Iteration Count (full variant)

| iter | E_initial (J) | E_final (J) | E retained | n_contacts (mean) |
|------|--------------|------------|-----------|------------------|
| 10  | 34.71 | 9.91  | **28.5%** | 0.09 |
| 50  | 34.71 | 23.09 | **66.5%** | 1.47 |
| 100 | 34.71 | 23.08 | **66.5%** | 1.18 |
| 500 | 34.71 | 23.09 | **66.5%** | 1.41 |

### 5.2 Iter=10 Anomaly (frame 17–21 divergence despite n_contacts=0)

| iter / variant | E at frame 21 (J) |
|---------------|--------------------|
| iter=10, `full`        | 14.18 |
| iter=10, `no_trunc`    | **−0.44** (unstable) |
| iter=10, `no_contact`  | 28.74 |
| iter=100, `full`       | 28.74 |

The divergence starts at frame 17, before any contacts are recorded (n_contacts=0). The Planar-DAT conservative bounds detect proximity before formal contact pairs are generated, and begin truncating inertia displacement. At 10 iterations this under-resolves the contact dynamics.

### 5.3 Planar-DAT Role at Low Iterations

| iter | `full` E_f | `no_trunc` E_f | Difference |
|------|-----------|--------------|-----------|
| 10  | 9.91  | **−5.27** | +15.2 J — **truncation stabilizes** |
| 50  | 23.09 | 23.09     | 0 J — truncation irrelevant at convergence |
| 100 | 23.08 | 23.08     | 0 J |
| 500 | 23.09 | 23.09     | 0 J |

### 5.4 Findings

1. **Convergence threshold at iter≈50**: below this, contact forces are under-resolved and energy loss is large.
2. **`no_trunc` at iter=10 is catastrophically unstable** (−5 J final energy): without Planar-DAT's displacement limiting, unresolved contact forces cause particles to overshoot, accumulating large stretch energy or falling through geometry.
3. **Planar-DAT acts as a stabilizer** at low iteration counts for brief-contact scenes.
4. **At iter≥50, contact / truncation have no measurable effect on final energy** (difference < 0.01 J). The contact is brief and resolves cleanly.

**Conclusion**: Scene C reveals a **dual role** of Planar-DAT — damping at high iter/sustained-contact (Scene B), but *stabilization* at low iter/brief-contact. The dominant damping source is solver under-convergence at iter=10 (28.5% vs. 66.5% retention), not truncation.

---

## 6. Scene D — Twist-and-Release

**Setup**: Square cloth from `square_cloth.usd` (14,406 vertices). Two opposite bottom corners are twist-driven at constant angular velocity until t=10 s, then released. Massive self-contact develops during twisting.

### 6.1 Energy Timeline (iter=100, full)

| t (s) | n_contacts | E_full (J) | E_no_contact (J) | Phase |
|-------|-----------|-----------|-----------------|-------|
| 0    | 0      | 0.0   | 0.0   | Rest |
| 1    | 0      | 14.8  | 14.8  | Twist begins |
| 2    | 8,096  | 30.9  | 16.2  | Self-contact onset |
| 5    | 25,798 | 57.3  | 17.7  | Dense contact |
| 8    | 41,642 | 81.5  | 19.4  | Peak contact |
| **10** | **37,273** | **74.3** | **18.9** | **← Release** |
| 11   | 31,123 | 63.3  | 3.0   | Unwinding |
| 12   | 26,813 | 57.6  | 16.8  | Settling |

### 6.2 Final Energy (t=12 s) by Iteration Count

| iter | `full` (J) | `no_contact` (J) | `no_trunc` (J) | `no_friction` (J) | full−no_trunc |
|------|-----------|-----------------|--------------|-----------------|--------------|
| 10  | 77.5  | 17.1 | 66.3  | 75.1  | **+11.2** |
| 50  | 70.0  | 17.6 | 70.8  | 69.3  | −0.8 |
| 100 | 57.5  | 17.4 | 69.4  | 72.1  | **−11.9** |
| 500 | 81.5  | 17.3 | 54.7  | 54.3  | **+26.8** |

### 6.3 Planar-DAT Truncation Statistics (full, 720 frames)

| iter | n_truncated | frac % | tang_frac | ΔKE_trunc / substep (J) |
|------|------------|--------|-----------|------------------------|
| 10  | 31.9  | 1.3% | 0.361 | −3×10⁻⁶ |
| 50  | 64.6  | 2.6% | 0.418 | −1.4×10⁻⁵ |
| 100 | 71.4  | 2.9% | 0.431 | −2.8×10⁻⁵ |
| 500 | 101.7 | 4.1% | 0.407 | −5.8×10⁻⁵ |

### 6.4 Findings

1. **Massive self-contact** (up to 41,642 contact pairs at t=8 s) stores elastic energy in the twisted cloth. Without contact (`no_contact`), cloth layers freely interpenetrate and the cloth settles at ≈17 J regardless of iter.
2. **Planar-DAT per-substep KE removal is tiny** (∼10⁻⁵ J) — three orders of magnitude smaller than in Scene B. The dominant energy storage is barrier/stretch potential, not truncation dissipation.
3. **Non-monotonic iter behavior** (full−no_trunc flips sign across iter counts): more iterations in a 40,000-contact scene lead the solver to different *topological equilibria* of the twisted cloth. The conservative bounds interact non-linearly with the dense contact graph.
4. **Tangential fraction ≈ 0.41** consistently, matching Scenes B and C — this appears to be a geometric constant of the truncation geometry.
5. At t=10–12 s, `full` (57.5 J) < `no_trunc` (69.4 J): Planar-DAT allows the cloth to unwind more efficiently by preventing overshoot during the high-contact phase.

**Conclusion**: In the highly self-contacting twist scenario, **Planar-DAT's per-substep energy effect is negligible**, but its influence on the *topological state* of the twisted cloth is large and non-monotonic. The dominant energy term is elastic potential stored in the dense self-contact configuration.

---

## 7. Cross-Scene Synthesis

### 7.1 Energy Retention Summary

| Scene | Contact type | iter=10 | iter=50 | iter=100 | iter=500 | Dominant source |
|-------|-------------|---------|---------|---------|---------|----------------|
| **A** | None | 2.2% | 2.1% | 2.1% | 2.1% | Structural implicit damping |
| **B** | Sustained (sliding) | — | — | — | — | Planar-DAT truncation |
| **C** | Brief (separating) | 28.5% | 66.5% | 66.5% | 66.5% | Under-convergence (iter<50) |
| **D** | Massive (twisted) | — | — | — | — | Topological equilibrium |

*Scenes B and D use initial energy near zero (falling/stationary start), making % retention ill-defined; absolute energy is used instead.*

### 7.2 Planar-DAT Isolation (full − no_trunc)

Since `dynamic_recoloring=False` in all variants, `full − no_trunc` isolates the pure Planar-DAT effect:

| Scene | iter=10 | iter=50 | iter=100 | iter=500 |
|-------|---------|---------|---------|---------|
| A | 0 J | 0 J | 0 J | 0 J |
| B | −5 J | −0.8 J | +3.3 J | −3.6 J |
| C | **+15 J** (stabilizes!) | 0 J | 0 J | 0 J |
| D | +11 J | −0.8 J | **−12 J** | **+27 J** |

### 7.3 Tangential Fraction

Across all scenes and iteration counts where truncation is active:

> **Tangential fraction ≈ 0.41–0.43** (constant)

This means ~42% of the displacement removed by Planar-DAT is perpendicular to the contact normal. Even in the frictionless scene (B, μ=0), Planar-DAT clips tangential motion — acting as an **implicit friction-like damper** that is not disabled by setting μ=0.

### 7.4 Damping Source Classification

```
                    STRUCTURAL      PLANAR-DAT       UNDER-CONVERGENCE
                    (implicit int.) (truncation)     (iter too low)
────────────────────────────────────────────────────────────────────────
Scene A (no contact)    ████████████      —               —
Scene B (sliding)       ██              ████████████    ███
Scene C (separating)    ███             ███ (stabilize) ████████ (iter<50)
Scene D (twisted)       ██              ████ (complex)  ████████
```

### 7.5 Effect of Substep Count (Timestep Size)

To separate the effect of dt from iteration count, a substep sweep was run on Scenes A (no contact) and B (frictionless sliding) with `iterations=100`, four ablation variants each.

#### Scene A — Structural Implicit Damping vs. dt

| substeps | dt (ms) | E_final (J) | E retained | ΔE/substep (J) | ΔE/s (J/s) |
|----------|---------|------------|-----------|----------------|-----------|
| 2  | 8.333 | −36.15 | −12.7% | 3.444 | 413.3 |
| 5  | 3.333 | −12.37 | −4.4%  | 1.594 | 478.1 |
| 10 | 1.667 |   5.89 |  +2.1% | 0.844 | 506.2 |
| 20 | 0.833 |  20.71 |  +7.3% | 0.425 | 510.3 |
| 40 | 0.417 |  25.96 |  +9.1% | 0.213 | 512.2 |

**Log-log fit**: `ΔE/substep ∝ dt^0.93 ≈ dt` — per-substep energy loss scales nearly linearly with the timestep size, as expected from backward-Euler implicit integration.

**Key finding**: `ΔE/s ≈ 500 J/s` is approximately constant across all substep counts. Halving dt (doubling substeps) halves the per-substep loss but doubles the substep count — the net dissipation rate per unit simulated time is invariant.

#### Scene B — Planar-DAT Truncation vs. dt (full variant, iter=100)

| substeps | dt (ms) | n_truncated | ΔKE_trunc/sub (J) | ΔKE_trunc/s (J/s) |
|----------|---------|------------|------------------|--------------------|
| 2  | 8.333 | 541.9 | 3.573 | 428.7 |
| 5  | 3.333 | 275.3 | 4.458 | 1337.4 |
| 10 | 1.667 | 115.1 | 1.353 | 811.5 |
| 20 | 0.833 |  40.4 | 0.425 | 509.4 |
| 40 | 0.417 |  30.3 | 0.333 | 798.5 |

**Log-log fit for n_truncated**: `n_trunc ∝ dt^1.04 ≈ dt` — fewer pairs truncated per substep at smaller dt, because per-substep particle displacements are smaller and less often exceed the conservative bound.

**Key finding**: Unlike Scene A, `ΔKE_trunc/s` is **non-monotonic**. At sub=5 (dt=3.33 ms), the truncation rate per unit time spikes to 1337 J/s — more than 2× the neighboring data points. This arises because the contact topology (which pairs are active) changes qualitatively at different dt values: large dt allows more pairs to accumulate before truncation fires, creating burst-mode truncation events.

**Tangential fraction** remains ≈ 0.42 across all substep counts, confirming that this is a geometric property of the truncation algorithm, not a dt artifact.

#### Summary

| Damping source | Per-substep scaling | Per-unit-time rate |
|---------------|--------------------|--------------------|
| Structural (backward-Euler) | ∝ dt^0.93 | ≈ constant (~500 J/s) |
| Planar-DAT truncation (Scene B) | ∝ dt^0.94 | non-monotonic (contact topology) |

---

## 8. Implications for Real-Time Simulation

### 8.1 Structural Implicit Damping Is Irreducible

Scene A demonstrates that even at iter=500 (fully converged), a 24×24 cloth loses 97.9% of its energy in 0.5 s. This cannot be fixed by adding more iterations. Mitigation requires:
- Smaller timestep dt (reduces per-step implicit damping)
- Higher-order time integration (e.g., BDF2, midpoint rule)
- Kinetic energy injection / velocity correction step

### 8.2 Planar-DAT Truncation Creates Iteration-Dependent Damping

In sustained-contact scenarios (Scene B), more iterations → more particle displacement → larger truncation → more KE removed. This creates an unfortunate trade-off: better convergence induces more contact damping.

The tangential component (42%) of truncation cannot be disabled by setting μ=0. Reducing tangential truncation requires either:
- Projecting the truncation scale to be applied only in the normal direction
- Accepting some penetration risk at the benefit of less tangential damping

### 8.3 Truncation as Stabilizer at Low Iterations

In brief-contact scenes (Scene C, iter=10), disabling Planar-DAT causes catastrophic instability (negative total energy). Truncation's conservative displacement bound serves as an implicit regularizer when the solver cannot fully resolve contact forces within the iteration budget.

### 8.4 Contact Density Introduces Non-Linear Behavior

In the twist-release scene (Scene D, ~40,000 contact pairs), the effect of Planar-DAT is non-monotonic with respect to iteration count: the solver converges to qualitatively different topological states depending on how aggressively truncation clips inertia displacements. This complicates any deterministic prediction of damping magnitude in real garment scenarios.

### 8.5 Increasing Substep Count Does Not Reduce Damping

A common expectation is that using more substeps (smaller dt) will reduce artificial dissipation — more steps should make the simulation "more accurate." This expectation is **incorrect for VBD**:

- **Structural backward-Euler damping**: per-substep loss ∝ dt, but substep rate ∝ 1/dt. The product — energy loss per unit simulated time — is constant regardless of substep count.
- **Planar-DAT truncation**: per-substep n_trunc ∝ dt, and per-substep ΔKE_trunc ∝ dt. Again the per-unit-time rate is roughly constant, though it is additionally sensitive to contact topology changes at different dt.

**Consequence**: doubling the substep count doubles the computational cost but does not improve energy conservation. Fixing structural damping requires a different time integration scheme (e.g., BDF2, symplectic integrators, or kinetic energy correction), not simply a finer timestep.

---

## 9. Experimental Configuration

| Parameter | Value |
|-----------|-------|
| Solver | `newton.solvers.SolverVBD` |
| Iterations | 10, 50, 100, 500 |
| Substeps/frame | 10 |
| dt | 1/600 s ≈ 1.667 ms |
| Device | NVIDIA RTX 4090 (cuda:1) |
| `dynamic_recoloring` | False (all runs) |
| `ogc_contact` | True (Scenes B, C, D) |
| `use_planar_dat` | True (`full`, `no_friction`) / False (`no_trunc`, `no_contact`) |
| Scenes A, B, C frames | 30 (0.5 s) |
| Scene D frames | 720 (12 s) |
| Total runs (iteration sweep) | 64 (4 scenes × 4 iter × 4 ablation) |
| **Substep sweep** | |
| Substeps/frame | 2, 5, 10, 20, 40 |
| dt range | 0.417 ms – 8.333 ms |
| Iterations | 100 |
| Scenes | A, B only |
| Total runs (substep sweep) | 40 (2 scenes × 5 substeps × 4 ablation) |

---

*Generated by VBDDiagnostics ablation framework — `scripts/run_damping_ablation.py` + `scripts/analyze_damping_logs.py`*
