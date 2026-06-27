"""
experiment_selfloop.py
----------------------
자기학습 루프를 합성 데이터로 돌려, 지도가 '자라는' 동안
추론경로(논리의 발자국)가 어떻게 변하는지 관측한다.

루프 (라운드마다):
  1. 현재 SOM에서 빈/저밀도 격자 영역을 찾는다 (= 호기심)
  2. 그 영역 근처에서 새 데이터를 '수집'한다 (합성: 해당 영역 중심 + 노이즈)
  3. SOM을 그 데이터로 온라인 학습 (가중치 갱신 = 성찰/심화)
  4. 고정된 질문/답변 임베딩쌍으로 추론경로를 측정 -> 지도 성장의 효과 관측
  5. 계측 로깅

두 조건 비교:
  GROW   : 빈 격자를 메우며 다양하게 성장 (경계 내 확장)
  ECHO   : 자기출력만 재섭취 (한 점 수렴, model collapse)
"""

import numpy as np
from som_introspect import SOMView, IntrospectLogger


def make_som(rows, cols, dim, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(scale=0.3, size=(rows, cols, dim))


def online_train(W, X, lr, radius):
    """간단한 SOM 온라인 학습. W를 in-place 갱신."""
    rows, cols, dim = W.shape
    flat = W.reshape(-1, dim)
    for x in X:
        d2 = np.einsum("nd,nd->n", flat - x, flat - x)
        bmu = int(np.argmin(d2))
        br, bc = divmod(bmu, cols)
        for r in range(rows):
            for c in range(cols):
                gd = (r - br) ** 2 + (c - bc) ** 2
                if gd <= radius ** 2:
                    h = np.exp(-gd / (2 * radius ** 2))
                    flat[r * cols + c] += lr * h * (x - flat[r * cols + c])
    return W


def low_density_centers(W, k=3):
    """가중치 norm이 작은(=덜 학습된) 격자 좌표 중심들을 호기심 타깃으로."""
    rows, cols, dim = W.shape
    flat = W.reshape(-1, dim)
    norms = np.linalg.norm(flat, axis=1)
    idx = np.argsort(norms)[:k]
    return flat[idx]


def run(condition: str, rounds=12, seed=0):
    rng = np.random.default_rng(seed)
    rows, cols, dim = 12, 12, 16
    W = make_som(rows, cols, dim, seed)
    logger = IntrospectLogger()

    # 고정된 질문/답변 임베딩쌍 (라운드 내내 동일) — 경로 변화를 보기 위해
    qa_anchor = [(rng.normal(size=dim), rng.normal(size=dim)) for _ in range(8)]

    echo_center = rng.normal(size=dim)

    for r in range(1, rounds + 1):
        if condition == "GROW":
            # 빈 격자 영역 근처에서 다양한 새 데이터 수집
            targets = low_density_centers(W, k=4)
            X = []
            for t in targets:
                X.append(t + rng.normal(scale=0.6, size=(10, dim)))
            X = np.vstack(X)
            vocab_pool = 20 + 4 * r          # 어휘 점점 풍부
        else:  # ECHO
            # 자기출력만 재섭취 — 한 점으로 수렴
            spread = max(0.03, 1.0 - 0.09 * r)
            X = echo_center[None, :] + rng.normal(scale=spread, size=(40, dim))
            vocab_pool = max(2, 24 - 2 * r)  # 어휘 점점 빈곤

        # 학습률/반경 감소 스케줄
        lr = 0.5 * (0.9 ** r)
        radius = max(1.0, 4.0 * (0.85 ** r))
        online_train(W, X, lr, radius)

        toks = [[f"w{int(v) % vocab_pool}" for v in (x[:8] * 5).astype(int)] for x in X]
        som = SOMView(W)
        m = logger.measure(
            round_idx=r, som=som, input_vectors=X,
            tokens_seen=toks, qa_pairs=qa_anchor,
        )
        warn = logger.collapse_warning()
        pc = f"{m.mean_path_cost:5.2f}" if m.mean_path_cost is not None else "  -  "
        ph = f"{m.mean_path_hops:4.1f}" if m.mean_path_hops is not None else "  - "
        print(f"[{condition}] r{r:2d} occ={m.occupancy:3d} qe={m.mean_qe:5.2f} "
              f"topoE={m.topo_error:.2f} vocab={m.vocab_div:.2f} "
              f"pathCost={pc} hops={ph}"
              + (f"  <<{warn.split(':')[0]}" if warn else ""))
    return logger


if __name__ == "__main__":
    print("=" * 70)
    g = run("GROW")
    print("  summary:", g.summary())
    print("=" * 70)
    e = run("ECHO")
    print("  summary:", e.summary())
    print("=" * 70)

    # 경로 지표만 따로 뽑아 출력 (차트용)
    print("\nROUND, GROW_pathCost, ECHO_pathCost, GROW_hops, ECHO_hops")
    for rg, re_ in zip(g.rows, e.rows):
        print(f"{rg.round}, "
              f"{rg.mean_path_cost:.3f}, {re_.mean_path_cost:.3f}, "
              f"{rg.mean_path_hops:.2f}, {re_.mean_path_hops:.2f}")
