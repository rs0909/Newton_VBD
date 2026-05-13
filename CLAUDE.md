@AGENTS.md
# CLAUDE.md

## Project context

This repository is for research on vertex block descent based real-time garment simulation.

The long-term research goal is:

> Real-time penetration-free garment simulation based on VBD.

Prioritize scientific correctness, minimal changes, and verifiable implementation over fast coding.

## Safety rules

- Do not modify files before proposing a plan.
- Do not implement large changes in one step.
- Prefer small, local changes.
- Do not silently change simulation semantics.
- Do not change time step, substep count, solver iterations, collision thresholds, mass, damping, units, or coordinate conventions without explicit approval.
- Do not overwrite datasets, experiment outputs, checkpoints, logs, or configs without approval.
- When uncertain, state assumptions instead of guessing.

## Implementation workflow

For each nontrivial task:

1. Inspect relevant files.
2. Summarize the current implementation.
3. Identify the insertion points.
4. Propose an implementation plan.
5. Wait for approval before editing.
6. Modify at most 2 files per implementation step unless approved.
7. After editing, summarize `git diff`.
8. Report risks and verification steps.

## Research correctness checklist

For algorithmic changes, always report:

- Mathematical formula being implemented.
- Tensor/array shapes.
- Units and coordinate assumptions.
- Whether the change affects parallelism, coloring, or solver ordering.
- Whether the change affects penetration guarantees.
- Numerical stability risks.
- GPU/CPU synchronization risks.
- Verification method.

## VBD / cloth simulation cautions

Be careful with:

- Graph coloring and same-color parallel updates.
- Per-substep versus per-iteration quantities.
- Stale geometric bounds.
- BVH query radius and pair filtering.
- Self-collision versus body-cloth collision.
- Mass-weighted updates.
- Double-push effects for same-color vertex pairs.
- Barrier stiffness and gradient scaling.
- Autograd is not required unless the existing project uses it.

## Language preference

Respond to the user in Korean. Keep code, comments, function names, variable names, config keys, log messages, commit messages, and terminal commands in English.