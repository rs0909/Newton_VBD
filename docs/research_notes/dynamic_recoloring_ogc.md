# Dynamic recoloring and same-color barrier for OGC/VBD

## Goal

The research goal is real-time penetration-free garment simulation based on vertex block descent.

The target failure case is the `cloth_twist_untwist` scene, where the garment can remain twisted and locked due to penetration.

## Background

The project is based on vertex block descent, where vertices in the same color group may be updated in parallel.

Offset Geometric Contact, OGC, computes a safe displacement bound for a vertex assuming other vertices are fixed.

A simplified OGC-style bound is:

\[
r_i^{safe} = \frac{d(x_i, nearest\ obstacle)}{2}
\]

This is conservative for single-vertex motion, but not necessarily conservative for simultaneous same-color vertex motion.

## Hypothesis

OGC may fail to prevent penetration in VBD for two reasons.

### 1. Simultaneous same-color motion

If vertices \(i\) and \(j\) are in the same color group, they may move toward each other simultaneously.

Each vertex can individually respect its safe bound, but the relative closing distance can be approximately the sum of both displacements.

Therefore, per-vertex safe bounds may not guarantee penetration-free behavior under same-color parallel updates.

### 2. Stale bounds

OGC safe bounds may be computed at the beginning of a substep and reused during VBD iterations.

As the configuration changes during iterations, the bound may become stale.

Increasing the number of iterations may increase the mismatch between the current geometry and the original bound.

## Proposed method

Introduce dynamic same-color risk handling.

For same-color vertex pairs:

1. If they are very close, dynamically recolor them to prevent simultaneous updates.
2. If they are not close enough for recoloring but still risky, add them to a watchlist.
3. During VBD iterations, evaluate only watchlist pairs and apply an IPC-like barrier if they become too close.

## Pair categories

For a same-color vertex pair \((i,j)\) with distance \(d_{ij}\):

### Recoloring region

If

\[
d_{ij} \le R_{recolor}(i,j)
\]

then the pair should be split into different colors.

This turns the update into a sequential solve and prevents same-color double-push.

### Watchlist region

If

\[
R_{recolor}(i,j) < d_{ij} \le R_{watchlist}(i,j)
\]

then the pair should be inserted into a watchlist.

During VBD iterations, if

\[
d_{ij} < \hat{d}
\]

apply a same-color barrier gradient.

### Ignore region

If

\[
d_{ij} > R_{watchlist}(i,j)
\]

ignore the pair and use the existing OGC behavior.

## Radius design

Reuse the OGC per-vertex substep displacement bound when possible.

Let \(r_i^{recolor}\) be the OGC substep movement bound for vertex \(i\).

Use:

\[
R_{recolor}(i,j) = r_i^{recolor} + r_j^{recolor}
\]

For the watchlist radius:

\[
r_i^{watchlist} = 2 r_i^{recolor}
\]

\[
R_{watchlist}(i,j) = r_i^{watchlist} + r_j^{watchlist}
\]

Implementation should reuse existing OGC bound data if available.

## Same-color barrier and double-push

Double-push can occur if both vertices in a same-color pair apply a full barrier response while treating the other vertex as fixed.

For equal masses, use half stiffness for each side:

\[
\nabla B_{same-color} = \frac{1}{2} \nabla B_{full}
\]

For unequal masses, use mass-weighted scaling.

For vertex \(i\):

\[
\alpha_i = \frac{m_j}{m_i + m_j}
\]

For vertex \(j\):

\[
\alpha_j = \frac{m_i}{m_i + m_j}
\]

The same-color barrier should be applied only for watchlist pairs, not recoloring pairs.

## Intended implementation stages

Do not implement everything at once.

Stage 1:
Analyze current code structure:
- VBD solver loop.
- Color groups and parallel update logic.
- OGC bound computation.
- BVH or neighbor query infrastructure.
- Collision/contact handling.
- Existing barrier or IPC-like terms.

Stage 2:
Identify where same-color pair detection could be added.

Stage 3:
Prototype watchlist construction only.
No solver behavior change yet.

Stage 4:
Prototype dynamic recoloring.
Keep it optional behind a config flag.

Stage 5:
Prototype same-color watchlist barrier.
Keep it optional behind a config flag.

Stage 6:
Add diagnostics for `cloth_twist_untwist`:
- number of recolored pairs,
- watchlist pair count,
- minimum same-color distance,
- penetration count/depth if available.

## Non-goals for initial implementation

- Do not rewrite the entire solver.
- Do not change the global VBD algorithm.
- Do not change unrelated collision handling.
- Do not change scene files or dataset files.
- Do not optimize performance before correctness.
- Do not merge step2 or messy branch code wholesale.