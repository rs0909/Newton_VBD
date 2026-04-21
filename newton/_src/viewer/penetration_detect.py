from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def compute_triangle_aabbs(v: np.ndarray, f: np.ndarray):
    tri = v[f]  # (T, 3, 3)
    bb_min = tri.min(axis=1)
    bb_max = tri.max(axis=1)
    return bb_min, bb_max


def are_adjacent(fi: np.ndarray, fj: np.ndarray) -> bool:
    # self-collision false positive 방지:
    # vertex 1개만 공유해도 제외할지, edge 공유만 제외할지는 선택 가능.
    # cloth는 보통 conservative하게 shared vertex 제외로 시작하는 게 안전.
    return len(set(fi.tolist()) & set(fj.tolist())) > 0


def aabb_overlap(a_min, a_max, b_min, b_max, eps=1e-8) -> bool:
    return np.all(a_min <= b_max + eps) and np.all(b_min <= a_max + eps)


def broad_phase_pairs(v: np.ndarray, f: np.ndarray) -> list[tuple[int, int]]:
    # 간단한 O(T^2) 버전. triangle 수가 많으면 BVH/spatial hash로 교체하세요.
    bb_min, bb_max = compute_triangle_aabbs(v, f)
    pairs: list[tuple[int, int]] = []
    T = len(f)
    for i in range(T):
        for j in range(i + 1, T):
            if not aabb_overlap(bb_min[i], bb_max[i], bb_min[j], bb_max[j]):
                continue
            if are_adjacent(f[i], f[j]):
                continue
            pairs.append((i, j))
    return pairs


def tri_tri_intersection_segment(
    tri_a: np.ndarray,
    tri_b: np.ndarray,
    eps: float = 1e-8,
) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
    """
    반환:
      (intersects, p0, p1)

    여기만 실제 narrow-phase 구현으로 교체하면 됩니다.
    production에서는 아래 셋 중 하나 추천:
      1) 기존 tri-tri exact predicate 구현 연결
      2) CCCollisions / torch-mesh-isect류 결과를 segment로 변환
      3) Warp/CUDA custom kernel
    """
    # TODO: robust tri-tri intersection 구현 연결
    # placeholder
    return False, None, None


@dataclass
class PenetrationDebugResult:
    pair_count: int
    triangle_count: int
    segment_count: int
    penetrating_triangles: np.ndarray
    segment_starts: np.ndarray
    segment_ends: np.ndarray
    triangle_centroids: np.ndarray


class SelfPenetrationDetector:
    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def compute(self, vertices: np.ndarray, faces: np.ndarray) -> PenetrationDebugResult:
        candidate_pairs = broad_phase_pairs(vertices, faces)

        seg_starts = []
        seg_ends = []
        tri_ids = set()

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
            tri_centroids = vertices[faces[penetrating_triangles]].mean(axis=1)
        else:
            tri_centroids = np.zeros((0, 3), dtype=np.float32)

        if len(seg_starts) == 0:
            seg_starts_np = np.zeros((0, 3), dtype=np.float32)
            seg_ends_np = np.zeros((0, 3), dtype=np.float32)
        else:
            seg_starts_np = np.asarray(seg_starts, dtype=np.float32)
            seg_ends_np = np.asarray(seg_ends, dtype=np.float32)

        return PenetrationDebugResult(
            pair_count=len(seg_starts),  # segment 1개 = pair 1개로 두는 단순 버전
            triangle_count=len(penetrating_triangles),
            segment_count=len(seg_starts_np),
            penetrating_triangles=penetrating_triangles,
            segment_starts=seg_starts_np,
            segment_ends=seg_ends_np,
            triangle_centroids=tri_centroids.astype(np.float32),
        )