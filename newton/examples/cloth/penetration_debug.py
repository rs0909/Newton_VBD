from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import warp as wp


@dataclass
class PenetrationDebugResult:
    pair_count: int
    triangle_count: int
    segment_count: int
    penetrating_triangles: np.ndarray   # (K,)
    segment_starts: np.ndarray          # (M, 3)
    segment_ends: np.ndarray            # (M, 3)
    triangle_centroids: np.ndarray      # (K, 3)


def are_adjacent(fi: np.ndarray, fj: np.ndarray) -> bool:
    return len(set(fi.tolist()) & set(fj.tolist())) > 0


def compute_triangle_aabbs(v: np.ndarray, f: np.ndarray):
    tri = v[f]  # (T, 3, 3)
    return tri.min(axis=1), tri.max(axis=1)


def aabb_overlap(a_min, a_max, b_min, b_max, eps=1e-8) -> bool:
    return np.all(a_min <= b_max + eps) and np.all(b_min <= a_max + eps)


def broad_phase_pairs(vertices: np.ndarray, faces: np.ndarray) -> list[tuple[int, int]]:
    bb_min, bb_max = compute_triangle_aabbs(vertices, faces)
    pairs: list[tuple[int, int]] = []
    T = len(faces)

    for i in range(T):
        for j in range(i + 1, T):
            if are_adjacent(faces[i], faces[j]):
                continue
            if not aabb_overlap(bb_min[i], bb_max[i], bb_min[j], bb_max[j]):
                continue
            pairs.append((i, j))
    return pairs


def tri_tri_intersection_segment(
    tri_a: np.ndarray,
    tri_b: np.ndarray,
    eps: float = 1e-8,
) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
    """
    TODO:
      여기만 robust tri-tri intersection 구현으로 교체.
      지금은 plumbing 확인용 placeholder.
    """
    return False, None, None


class SelfPenetrationDetector:
    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def compute(self, vertices: np.ndarray, faces: np.ndarray) -> PenetrationDebugResult:
        candidate_pairs = broad_phase_pairs(vertices, faces)

        tri_ids = set()
        seg_starts = []
        seg_ends = []

        for ti, tj in candidate_pairs:
            tri_a = vertices[faces[ti]]
            tri_b = vertices[faces[tj]]

            hit, p0, p1 = tri_tri_intersection_segment(tri_a, tri_b, self.eps)
            if not hit:
                continue

            tri_ids.add(ti)
            tri_ids.add(tj)

            if p0 is not None and p1 is not None:
                seg_starts.append(p0)
                seg_ends.append(p1)

        penetrating_triangles = np.array(sorted(tri_ids), dtype=np.int32)

        if len(penetrating_triangles) > 0:
            triangle_centroids = vertices[faces[penetrating_triangles]].mean(axis=1).astype(np.float32)
        else:
            triangle_centroids = np.zeros((0, 3), dtype=np.float32)

        if len(seg_starts) > 0:
            segment_starts = np.asarray(seg_starts, dtype=np.float32)
            segment_ends = np.asarray(seg_ends, dtype=np.float32)
        else:
            segment_starts = np.zeros((0, 3), dtype=np.float32)
            segment_ends = np.zeros((0, 3), dtype=np.float32)

        return PenetrationDebugResult(
            pair_count=len(seg_starts),
            triangle_count=len(penetrating_triangles),
            segment_count=len(segment_starts),
            penetrating_triangles=penetrating_triangles,
            segment_starts=segment_starts,
            segment_ends=segment_ends,
            triangle_centroids=triangle_centroids,
        )


class PenetrationViewerDebug:
    def __init__(self, viewer, device="cpu"):
        self.viewer = viewer
        self.device = wp.get_device(device) if isinstance(device, str) else device
        self.latest = {
            "pair_count": 0,
            "triangle_count": 0,
            "segment_count": 0,
        }

        if hasattr(viewer, "register_ui_callback"):
            viewer.register_ui_callback(self._render_ui, position="stats")

    def _render_ui(self, imgui):
        if imgui.collapsing_header("Penetration Debug"):
            imgui.text(f"Intersecting pairs: {self.latest['pair_count']}")
            imgui.text(f"Penetrating triangles: {self.latest['triangle_count']}")
            imgui.text(f"Intersection segments: {self.latest['segment_count']}")

    def _to_wp_vec3(self, arr: np.ndarray):
        arr = np.asarray(arr, dtype=np.float32)
        return wp.array(arr, dtype=wp.vec3, device=self.device)

    def _make_color_array(self, n: int, rgb: tuple[float, float, float]):
        colors = np.tile(np.array(rgb, dtype=np.float32)[None, :], (n, 1))
        return wp.array(colors, dtype=wp.vec3, device=self.device)

    def log(self, result: PenetrationDebugResult):
        self.latest["pair_count"] = int(result.pair_count)
        self.latest["triangle_count"] = int(result.triangle_count)
        self.latest["segment_count"] = int(result.segment_count)

        empty_vec3 = wp.array(
            np.zeros((0, 3), dtype=np.float32),
            dtype=wp.vec3,
            device=self.device,
        )

        # segments
        if result.segment_count > 0:
            seg_starts = self._to_wp_vec3(result.segment_starts)
            seg_ends = self._to_wp_vec3(result.segment_ends)
            seg_colors = self._make_color_array(result.segment_count, (1.0, 0.1, 0.1))
            hidden_segments = False
        else:
            seg_starts = empty_vec3
            seg_ends = empty_vec3
            seg_colors = empty_vec3
            hidden_segments = True

        self.viewer.log_lines(
            "debug/penetration/segments",
            seg_starts,
            seg_ends,
            colors=seg_colors,
            width=0.01,
            hidden=hidden_segments,
        )

        # centroids
        if result.triangle_count > 0:
            centroids = self._to_wp_vec3(result.triangle_centroids)
            point_colors = self._make_color_array(result.triangle_count, (1.0, 0.9, 0.1))
            hidden_points = False
        else:
            centroids = empty_vec3
            point_colors = empty_vec3
            hidden_points = True

        self.viewer.log_points(
            "debug/penetration/centroids",
            centroids,
            radii=0.01,
            colors=point_colors,
            hidden=hidden_points,
        )

        self.viewer.log_scalar("penetration/intersecting_pairs", result.pair_count)
        self.viewer.log_scalar("penetration/penetrating_triangles", result.triangle_count)
        self.viewer.log_scalar("penetration/intersection_segments", result.segment_count)

    def __init__(self, viewer, device=None):
        self.viewer = viewer
        self.device = device or wp.get_device()
        self.latest = {
            "pair_count": 0,
            "triangle_count": 0,
            "segment_count": 0,
        }

        if hasattr(viewer, "register_ui_callback"):
            viewer.register_ui_callback(self._render_ui, position="stats")

    def _render_ui(self, imgui):
        if imgui.collapsing_header("Penetration Debug"):
            imgui.text(f"Intersecting pairs: {self.latest['pair_count']}")
            imgui.text(f"Penetrating triangles: {self.latest['triangle_count']}")
            imgui.text(f"Intersection segments: {self.latest['segment_count']}")

    def _to_wp_vec3(self, arr: np.ndarray):
        return wp.array(arr.astype(np.float32), dtype=wp.vec3, device=self.device)

    def log(self, result: PenetrationDebugResult):
        self.latest["pair_count"] = int(result.pair_count)
        self.latest["triangle_count"] = int(result.triangle_count)
        self.latest["segment_count"] = int(result.segment_count)

        empty_vec3 = wp.array(
            np.zeros((0, 3), dtype=np.float32),
            dtype=wp.vec3,
            device=self.device,
        )

        # segments
        if result.segment_count > 0:
            seg_starts = self._to_wp_vec3(result.segment_starts)
            seg_ends = self._to_wp_vec3(result.segment_ends)
            seg_colors = self._make_color_array(result.segment_count, (1.0, 0.1, 0.1))
            hidden_segments = False
        else:
            seg_starts = empty_vec3
            seg_ends = empty_vec3
            seg_colors = empty_vec3
            hidden_segments = True

        self.viewer.log_lines(
            "debug/penetration/segments",
            seg_starts,
            seg_ends,
            colors=seg_colors,
            width=0.01,
            hidden=hidden_segments,
        )

        # centroids
        if result.triangle_count > 0:
            centroids = self._to_wp_vec3(result.triangle_centroids)
            point_colors = self._make_color_array(result.triangle_count, (1.0, 0.9, 0.1))
            hidden_points = False
        else:
            centroids = empty_vec3
            point_colors = empty_vec3
            hidden_points = True

        self.viewer.log_points(
            "debug/penetration/centroids",
            centroids,
            radii=0.01,
            colors=point_colors,
            hidden=hidden_points,
        )

        self.viewer.log_scalar("penetration/intersecting_pairs", result.pair_count)
        self.viewer.log_scalar("penetration/penetrating_triangles", result.triangle_count)
        self.viewer.log_scalar("penetration/intersection_segments", result.segment_count)