"""
som_introspect.py
------------------
자기학습 루프(self-reflection loop)가 '고도화'로 가는지 '붕괴(collapse)'로
가는지를 SOM 지표로 관측하기 위한 계측 모듈.

설계 원칙
- 기존 gascore_engine.py 를 수정하지 않는다. SOM 가중치 행렬과 입력 벡터만 받는다.
- 매 라운드 핵심 지표를 기록한다:
    occupancy   : 격자 점유율 (몇 개 격자가 BMU로 쓰였나)  -> 줄면 붕괴 징후
    empty_cells : 빈 격자 수                                -> 0으로 수렴하면 다양성 소멸
    mean_qe     : 평균 양자화오차(입력<->BMU 거리)          -> 계속 줄기만 하면 메아리방
    topo_error  : 토포그래픽 오차(1등2등 BMU 인접 실패율)   -> 늘면 지도가 찢어짐
    vocab_div   : 어휘 다양성(서로 다른 토큰/전체)          -> 줄면 말이 좁아짐(붕괴)
    path_len    : 질문격자->답변격자 추론경로 길이(선택)    -> '논리의 발자국'

지표 해석 (collapse vs deepening)
    붕괴:  occupancy↓  empty_cells↓(한점수렴)  mean_qe↓(새것없음)  vocab_div↓
    고도화: occupancy↑/유지  mean_qe 안정  topo_error 낮음 유지  vocab_div 유지/↑
"""

from __future__ import annotations
import json
import math
import time
from dataclasses import dataclass, asdict, field
from typing import Sequence, Optional

import numpy as np


# ----------------------------------------------------------------------
# SOM 어댑터: 기존 엔진의 가중치 행렬만 있으면 동작
# ----------------------------------------------------------------------
class SOMView:
    """
    기존 SOM의 가중치만 감싸는 읽기 전용 뷰.
    weights shape = (rows, cols, dim)  -- 표준 2D SOM 격자.
    gascore_engine.py 의 SOM이 1D(nodes, dim)이면 rows=1로 넣으면 된다.
    """

    def __init__(self, weights: np.ndarray):
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim == 2:                      # (nodes, dim) -> (1, nodes, dim)
            w = w[None, :, :]
        assert w.ndim == 3, "weights must be (rows, cols, dim) or (nodes, dim)"
        self.w = w
        self.rows, self.cols, self.dim = w.shape
        self.flat = w.reshape(-1, self.dim)  # (rows*cols, dim)

    def _dists(self, x: np.ndarray) -> np.ndarray:
        # x: (dim,) -> 각 노드까지 유클리드 거리^2
        diff = self.flat - x[None, :]
        return np.einsum("nd,nd->n", diff, diff)

    def bmu(self, x: np.ndarray) -> tuple[int, float]:
        """best matching unit index(flat), 그리고 거리(=양자화오차)."""
        d2 = self._dists(x)
        i = int(np.argmin(d2))
        return i, math.sqrt(float(d2[i]))

    def bmu2(self, x: np.ndarray) -> tuple[int, int]:
        """1등, 2등 BMU의 flat index."""
        d2 = self._dists(x)
        order = np.argpartition(d2, 2)[:2]
        order = order[np.argsort(d2[order])]
        return int(order[0]), int(order[1])

    def coord(self, flat_idx: int) -> tuple[int, int]:
        return divmod(flat_idx, self.cols)

    def are_neighbors(self, a: int, b: int) -> bool:
        ra, ca = self.coord(a)
        rb, cb = self.coord(b)
        return max(abs(ra - rb), abs(ca - cb)) <= 1  # 8-이웃(체비셰프)

    def _neighbors(self, flat_idx: int):
        """8-이웃 격자의 flat index 리스트."""
        r, c = self.coord(flat_idx)
        out = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    out.append(nr * self.cols + nc)
        return out

    def _edge_cost(self, a: int, b: int) -> float:
        """이웃 격자 간 이동비용 = 가중치 벡터 유클리드 거리(=의미 거리)."""
        d = self.flat[a] - self.flat[b]
        return math.sqrt(float(d @ d))

    def semantic_path(self, src: int, dst: int):
        """
        의미 가중 최단경로(Dijkstra).
        반환: (path[list of flat idx], total_cost, hop_len)
        - total_cost : 의미 거리 합 (질문->답변 '의미적 도정 거리')
        - hop_len    : 거친 격자 수 - 1 (몇 칸 밟았나)
        """
        import heapq
        if src == dst:
            return [src], 0.0, 0
        N = self.rows * self.cols
        INF = float("inf")
        dist = [INF] * N
        prev = [-1] * N
        dist[src] = 0.0
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            if u == dst:
                break
            for v in self._neighbors(u):
                nd = d + self._edge_cost(u, v)
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        if dist[dst] == INF:
            return [src, dst], INF, -1
        # 경로 복원
        path = []
        cur = dst
        while cur != -1:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return path, dist[dst], len(path) - 1

    def reasoning_path(self, q_vec: np.ndarray, a_vec: np.ndarray):
        """질문 임베딩->답변 임베딩의 의미 추론 궤적."""
        qi, _ = self.bmu(np.asarray(q_vec, dtype=np.float64))
        ai, _ = self.bmu(np.asarray(a_vec, dtype=np.float64))
        return self.semantic_path(qi, ai)


# ----------------------------------------------------------------------
# 라운드 1회 측정 결과
# ----------------------------------------------------------------------
@dataclass
class RoundMetrics:
    round: int
    n_inputs: int
    occupancy: int          # BMU로 한번이라도 선택된 격자 수
    occupancy_ratio: float  # occupancy / 전체격자
    empty_cells: int
    mean_qe: float
    topo_error: float
    vocab_div: float
    path_len: Optional[float] = None       # (구) 외부 제공 경로 hop 합
    mean_path_cost: Optional[float] = None  # 질문->답변 평균 '의미 도정거리'
    mean_path_hops: Optional[float] = None  # 질문->답변 평균 격자 hop 수
    wall_time_s: Optional[float] = None

    def as_row(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------
# 계측기 본체
# ----------------------------------------------------------------------
class IntrospectLogger:
    def __init__(self):
        self.rows: list[RoundMetrics] = []

    def measure(
        self,
        round_idx: int,
        som: SOMView,
        input_vectors: Sequence[np.ndarray],
        tokens_seen: Optional[Sequence[Sequence[str]]] = None,
        reasoning_path: Optional[Sequence[int]] = None,
        qa_pairs: Optional[Sequence[tuple]] = None,
        wall_time_s: Optional[float] = None,
    ) -> RoundMetrics:
        """
        round_idx      : 라운드 번호(시간옵션과 무관하게 재현용 축)
        som            : SOMView
        input_vectors  : 이번 라운드에 지도에 들어온 임베딩들 (각 (dim,))
        tokens_seen    : 어휘다양성 계산용. 각 입력의 토큰 리스트.
        reasoning_path : (구) 외부에서 직접 준 flat index 궤적(선택)
        qa_pairs       : [(q_vec, a_vec), ...] 질문/답변 임베딩쌍.
                         주면 의미 가중 최단경로로 추론궤적을 자동 계산.
        """
        X = [np.asarray(v, dtype=np.float64) for v in input_vectors]
        n = len(X)

        bmus, qes = [], []
        topo_fail = 0
        for x in X:
            i, qe = som.bmu(x)
            bmus.append(i)
            qes.append(qe)
            b1, b2 = som.bmu2(x)
            if not som.are_neighbors(b1, b2):
                topo_fail += 1

        total_cells = som.rows * som.cols
        occ = len(set(bmus))
        empty = total_cells - occ
        mean_qe = float(np.mean(qes)) if qes else 0.0
        topo_err = (topo_fail / n) if n else 0.0

        # 어휘 다양성 (type-token ratio)
        vocab_div = 0.0
        if tokens_seen:
            all_tok = [t for seq in tokens_seen for t in seq]
            if all_tok:
                vocab_div = len(set(all_tok)) / len(all_tok)

        # 추론경로 길이: 인접 이동 거리 합(체비셰프)
        path_len = None
        if reasoning_path and len(reasoning_path) >= 2:
            steps = 0.0
            for a, b in zip(reasoning_path[:-1], reasoning_path[1:]):
                ra, ca = som.coord(a)
                rb, cb = som.coord(b)
                steps += max(abs(ra - rb), abs(ca - cb))
            path_len = steps

        # 추론경로 (의미 가중 최단경로)
        mean_path_cost = None
        mean_path_hops = None
        if qa_pairs:
            costs, hops = [], []
            for qv, av in qa_pairs:
                _, cost, hop = som.reasoning_path(qv, av)
                if math.isfinite(cost):
                    costs.append(cost)
                    hops.append(hop)
            if costs:
                mean_path_cost = float(np.mean(costs))
                mean_path_hops = float(np.mean(hops))

        m = RoundMetrics(
            round=round_idx,
            n_inputs=n,
            occupancy=occ,
            occupancy_ratio=occ / total_cells if total_cells else 0.0,
            empty_cells=empty,
            mean_qe=mean_qe,
            topo_error=topo_err,
            vocab_div=vocab_div,
            path_len=path_len,
            mean_path_cost=mean_path_cost,
            mean_path_hops=mean_path_hops,
            wall_time_s=wall_time_s,
        )
        self.rows.append(m)
        return m

    # ---- 붕괴 조기경보 ----
    def collapse_warning(self, window: int = 3) -> Optional[str]:
        """
        최근 window 라운드 추세로 붕괴 징후를 판정.
        occupancy와 vocab_div가 동반 단조감소 + mean_qe도 감소 -> 메아리방.
        """
        if len(self.rows) < window + 1:
            return None
        recent = self.rows[-(window + 1):]
        occ = [r.occupancy for r in recent]
        voc = [r.vocab_div for r in recent]
        qe = [r.mean_qe for r in recent]

        def mono_down(seq):
            return all(b <= a for a, b in zip(seq[:-1], seq[1:])) and seq[0] > seq[-1]

        if mono_down(occ) and mono_down(voc) and mono_down(qe):
            return ("COLLAPSE WARNING: 점유율·어휘다양성·QE 동반 감소. "
                    "자기출력 메아리방(model collapse) 진입 가능성.")
        if mono_down(occ) and recent[-1].occupancy_ratio < 0.1:
            return ("COLLAPSE WARNING: 점유율이 10% 미만으로 한 점 수렴 중.")
        return None

    # ---- 저장 ----
    def to_jsonl(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            for r in self.rows:
                f.write(json.dumps(r.as_row(), ensure_ascii=False) + "\n")

    def summary(self) -> dict:
        if not self.rows:
            return {}
        first, last = self.rows[0], self.rows[-1]
        return {
            "rounds": len(self.rows),
            "occupancy": [first.occupancy, last.occupancy],
            "empty_cells": [first.empty_cells, last.empty_cells],
            "mean_qe": [round(first.mean_qe, 4), round(last.mean_qe, 4)],
            "topo_error": [round(first.topo_error, 4), round(last.topo_error, 4)],
            "vocab_div": [round(first.vocab_div, 4), round(last.vocab_div, 4)],
            "trend": "deepening" if (last.occupancy >= first.occupancy
                                     and last.vocab_div >= first.vocab_div * 0.9)
                     else "collapse-risk",
        }


# ----------------------------------------------------------------------
# 데모: 두 시나리오를 합성 데이터로 돌려 지표가 갈리는지 검증
# ----------------------------------------------------------------------
def _demo():
    rng = np.random.default_rng(0)
    dim = 16
    grid = (10, 10)

    # 임의 SOM 가중치(고정) — 실제로는 gascore_engine SOM.weights 를 넣는다
    W = rng.normal(size=(*grid, dim))
    som = SOMView(W)

    print("=== 시나리오 A: 고도화 (입력이 점점 다양·확장) ===")
    logA = IntrospectLogger()
    for r in range(1, 11):
        # 라운드가 갈수록 더 넓은 영역을 덮는 입력
        spread = 0.5 + 0.3 * r
        X = rng.normal(scale=spread, size=(40, dim))
        toks = [[f"w{int(v)}" for v in (x[:8] * 5).astype(int)] for x in X]
        m = logA.measure(r, som, X, tokens_seen=toks)
        warn = logA.collapse_warning()
        print(f"  r{r:2d} occ={m.occupancy:3d} empty={m.empty_cells:3d} "
              f"qe={m.mean_qe:5.2f} topoE={m.topo_error:.2f} vocab={m.vocab_div:.2f}"
              + (f"  <-- {warn}" if warn else ""))
    print("  summary:", logA.summary())

    print("\n=== 시나리오 B: 붕괴 (입력이 한 점으로 수렴=자기출력 재섭취) ===")
    logB = IntrospectLogger()
    center = rng.normal(size=dim)
    for r in range(1, 11):
        # 라운드가 갈수록 분산이 줄어 한 점으로 — 메아리방
        spread = max(0.02, 1.2 - 0.13 * r)
        X = center[None, :] + rng.normal(scale=spread, size=(40, dim))
        # 어휘도 같이 좁아짐
        vocab_pool = max(2, 30 - 3 * r)
        toks = [[f"w{int(v) % vocab_pool}" for v in (x[:8] * 5).astype(int)] for x in X]
        m = logB.measure(r, som, X, tokens_seen=toks)
        warn = logB.collapse_warning()
        print(f"  r{r:2d} occ={m.occupancy:3d} empty={m.empty_cells:3d} "
              f"qe={m.mean_qe:5.2f} topoE={m.topo_error:.2f} vocab={m.vocab_div:.2f}"
              + (f"  <-- {warn}" if warn else ""))
    print("  summary:", logB.summary())

    return logA, logB


if __name__ == "__main__":
    _demo()
