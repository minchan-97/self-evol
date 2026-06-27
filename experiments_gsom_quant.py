"""
gsom_quant.py
-------------
심화 SOM 실험 3종 결합:
  1) 실제 한국어 문장 임베딩 투입 (load_embeddings 만 갈아끼우면 TinyTransformer 연결)
  2) Growing SOM (GSOM) — 빈/고오차 영역에서 노드를 늘려 스스로 자람
  3) INT8 양자화 — SOM 가중치를 저장/연산 시 압축해 속도·용량 보완

양자화 핵심 (정직한 설계):
  - 저장/전송: INT8 (4배 압축)
  - BMU 탐색: INT8 정수공간에서 직접 거리 계산 (역양자화 없이) -> 속도↑
  - scale/zero_point 만 FP32로 보관
"""

from __future__ import annotations
import time
import hashlib
import numpy as np


# ======================================================================
# 1. 임베딩 로더 — 여기만 갈아끼우면 실제 TinyTransformer tok_emb 연결됨
# ======================================================================
def load_embeddings(sentences, dim=64, seed=42):
    """
    결정론적 해시 임베딩 (네트워크 불필요, 재현 가능).
    실제 사용 시: 이 함수 본문을 TinyTransformer 임베딩 호출로 교체.
        return np.stack([tok_emb_encode(s) for s in sentences])
    같은 문장 -> 같은 벡터, 비슷한 문장(공유 어절) -> 가까운 벡터가 되도록
    어절 단위 해시 임베딩을 평균.
    """
    rng_base = np.random.default_rng(seed)
    # 어절 -> 고정 벡터 사전(요청 시 생성)
    cache: dict[str, np.ndarray] = {}

    def word_vec(w: str) -> np.ndarray:
        if w not in cache:
            h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
            r = np.random.default_rng(h % (2**32))
            cache[w] = r.normal(size=dim)
        return cache[w]

    vecs = []
    for s in sentences:
        words = s.split()
        if not words:
            vecs.append(rng_base.normal(size=dim))
            continue
        v = np.mean([word_vec(w) for w in words], axis=0)
        vecs.append(v)
    X = np.asarray(vecs, dtype=np.float64)
    # 정규화
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    return X


# ======================================================================
# 2. INT8 양자화기
# ======================================================================
class Quantizer:
    """대칭 선형 INT8 양자화. 채널(차원)별 scale."""

    def __init__(self, W: np.ndarray):
        # W: (n_nodes, dim)
        self.absmax = np.abs(W).max(axis=0) + 1e-8     # (dim,)
        self.scale = self.absmax / 127.0               # (dim,)

    def q(self, W: np.ndarray) -> np.ndarray:
        return np.clip(np.round(W / self.scale), -127, 127).astype(np.int8)

    def dq(self, Wq: np.ndarray) -> np.ndarray:
        return Wq.astype(np.float64) * self.scale

    def q_vec(self, x: np.ndarray) -> np.ndarray:
        return np.clip(np.round(x / self.scale), -127, 127).astype(np.int8)


def bmu_int8(Wq: np.int8, xq: np.int8) -> tuple[int, int]:
    """INT8 정수공간에서 직접 BMU 탐색 (역양자화 없이). 거리^2(int32) 반환."""
    diff = Wq.astype(np.int32) - xq.astype(np.int32)[None, :]
    d2 = np.einsum("nd,nd->n", diff, diff)
    i = int(np.argmin(d2))
    return i, int(d2[i])


# ======================================================================
# 3. Growing SOM
# ======================================================================
class GrowingSOM:
    def __init__(self, dim, init_nodes=16, grow_threshold=2.5, seed=0):
        rng = np.random.default_rng(seed)
        self.dim = dim
        self.W = rng.normal(scale=0.3, size=(init_nodes, dim))
        # 노드 좌표(2D 평면에 배치, 성장 시 근처에 삽입)
        self.coords = rng.normal(scale=1.0, size=(init_nodes, 2))
        self.grow_threshold = grow_threshold
        self.err = np.zeros(init_nodes)   # 노드별 누적 오차

    @property
    def n(self):
        return self.W.shape[0]

    def bmu(self, x):
        d2 = np.einsum("nd,nd->n", self.W - x, self.W - x)
        i = int(np.argmin(d2))
        return i, float(np.sqrt(d2[i]))

    def train_step(self, X, lr, radius):
        for x in X:
            i, dist = self.bmu(x)
            self.err[i] += dist
            # 좌표 거리 기반 이웃 갱신
            cd2 = np.einsum("nd,nd->n", self.coords - self.coords[i],
                            self.coords - self.coords[i])
            h = np.exp(-cd2 / (2 * radius ** 2))
            self.W += (lr * h)[:, None] * (x - self.W)

    def grow(self, max_add=4):
        """누적오차 큰 노드 근처에 새 노드를 삽입 (= 그 영역을 더 정밀하게)."""
        added = 0
        order = np.argsort(self.err)[::-1]
        for i in order[:max_add]:
            if self.err[i] < self.grow_threshold:
                break
            # 부모 노드 근처에 새 노드: 가중치는 부모+노이즈, 좌표는 부모 옆
            new_w = self.W[i] + np.random.default_rng(self.n).normal(scale=0.1, size=self.dim)
            new_c = self.coords[i] + np.random.default_rng(self.n + 1).normal(scale=0.3, size=2)
            self.W = np.vstack([self.W, new_w])
            self.coords = np.vstack([self.coords, new_c])
            self.err = np.append(self.err, 0.0)
            self.err[i] *= 0.5   # 부모 오차 절반 분배
            added += 1
        return added

    def reset_err(self):
        self.err *= 0.0


# ======================================================================
# 4. 실험: 실제 문장 + GSOM 성장 + 양자화 속도/용량 측정
# ======================================================================
KO_SENTENCES = [
    # easy-read 스타일 (쉬운 문장)
    "오늘 날씨가 좋아요", "밥을 먹어요", "학교에 가요", "친구를 만나요",
    "책을 읽어요", "손을 씻어요", "물을 마셔요", "잠을 자요",
    "버스를 타요", "노래를 불러요", "그림을 그려요", "운동을 해요",
    # 도메인 (발달장애 교육)
    "선생님이 도와줘요", "천천히 말해요", "다시 해 봐요", "잘 했어요",
    "차례를 지켜요", "기다려 주세요", "함께 놀아요", "규칙을 지켜요",
    # 형식체 (FATAL 대상 — 어려운 문장)
    "본 계약은 당사자 간 합의에 의하여 체결된다",
    "해당 조항은 관련 법령에 근거하여 적용된다",
    "투자 수익률은 시장 변동성에 따라 결정된다",
    "양자화 오차는 신호 대 잡음비로 측정된다",
]


def run_experiment(rounds=10, grow=True, quantize=True):
    dim = 64
    X_all = load_embeddings(KO_SENTENCES, dim=dim)
    n_sent = len(KO_SENTENCES)

    gsom = GrowingSOM(dim=dim, init_nodes=16, grow_threshold=2.0, seed=1)

    # 고정 질문/답변쌍 (추론경로 관측용): 쉬운질문 -> 쉬운답변
    qa_idx = [(2, 0), (3, 6), (1, 5), (8, 10)]  # 문장 인덱스쌍

    print(f"{'R':>2} {'nodes':>5} {'qe_fp32':>8} {'qe_int8':>8} "
          f"{'t_fp32(ms)':>10} {'t_int8(ms)':>10} {'mem_fp32':>9} {'mem_int8':>9} {'grew':>4}")

    log = []
    for r in range(1, rounds + 1):
        lr = 0.4 * (0.9 ** r)
        radius = max(0.5, 2.0 * (0.85 ** r))
        gsom.reset_err()
        # 매 라운드 전체 문장 투입 (실제로는 크롤링 신규문장)
        gsom.train_step(X_all, lr, radius)

        grew = gsom.grow(max_add=4) if grow else 0

        # ---- 양자화 속도/정확도/용량 비교 ----
        W = gsom.W
        quant = Quantizer(W)
        Wq = quant.q(W)

        # FP32 BMU 속도 & 평균 QE
        t0 = time.perf_counter()
        qe_fp = []
        for x in X_all:
            d2 = np.einsum("nd,nd->n", W - x, W - x)
            qe_fp.append(np.sqrt(d2.min()))
        t_fp = (time.perf_counter() - t0) * 1000

        # INT8 BMU 속도 & 평균 QE (정수공간 직접 계산)
        t0 = time.perf_counter()
        qe_iq = []
        for x in X_all:
            xq = quant.q_vec(x)
            i, d2 = bmu_int8(Wq, xq)
            # 비교용 실제거리는 dq로 환산
            qe_iq.append(np.linalg.norm(quant.dq(Wq[i]) - x))
        t_iq = (time.perf_counter() - t0) * 1000

        mem_fp = W.nbytes
        mem_iq = Wq.nbytes + quant.scale.nbytes

        print(f"{r:>2} {gsom.n:>5} {np.mean(qe_fp):>8.3f} {np.mean(qe_iq):>8.3f} "
              f"{t_fp:>10.2f} {t_iq:>10.2f} {mem_fp:>9} {mem_iq:>9} {grew:>4}")

        log.append({
            "round": r, "nodes": gsom.n,
            "qe_fp": float(np.mean(qe_fp)), "qe_iq": float(np.mean(qe_iq)),
            "t_fp": t_fp, "t_iq": t_iq, "mem_fp": mem_fp, "mem_iq": mem_iq,
            "grew": grew,
        })

    # ---- 추론경로: 최종 지도에서 의미 도정거리 ----
    from som_introspect import SOMView
    sv = SOMView(gsom.W)   # 1D 노드 집합 -> (1, n, dim)로 처리됨
    print("\n[추론경로 @ 최종지도]  (질문문장 -> 답변문장)")
    for qi, ai in qa_idx:
        path, cost, hops = sv.semantic_path(*sv.bmu(X_all[qi])[:1] + sv.bmu(X_all[ai])[:1])
        # bmu가 (idx,dist) 튜플이라 위 호출이 어색 -> 직접 분해
    # 위 한 줄이 지저분하므로 명시적으로 다시
    print("  (아래 표 참조)")
    rows = []
    for qi, ai in qa_idx:
        qbmu, _ = sv.bmu(X_all[qi])
        abmu, _ = sv.bmu(X_all[ai])
        path, cost, hops = sv.semantic_path(qbmu, abmu)
        rows.append((KO_SENTENCES[qi], KO_SENTENCES[ai], cost, hops))
        print(f"   '{KO_SENTENCES[qi]}' -> '{KO_SENTENCES[ai]}'  "
              f"cost={cost:.2f} hops={hops}")

    return log, rows


if __name__ == "__main__":
    print("=" * 95)
    print("GROWING SOM + INT8 QUANTIZATION + 실제(해시)임베딩")
    print("=" * 95)
    log, rows = run_experiment(rounds=10, grow=True, quantize=True)

    print("\n" + "=" * 95)
    print("요약")
    print("=" * 95)
    first, last = log[0], log[-1]
    print(f"노드 성장:      {first['nodes']} -> {last['nodes']}")
    print(f"INT8 속도:      FP32 대비 평균 "
          f"{np.mean([l['t_fp']/l['t_iq'] for l in log]):.2f}x")
    print(f"INT8 용량:      FP32 대비 "
          f"{last['mem_fp']/last['mem_iq']:.2f}x 압축")
    print(f"INT8 QE 오차:   FP32 대비 평균 "
          f"{np.mean([abs(l['qe_iq']-l['qe_fp'])/l['qe_fp'] for l in log])*100:.2f}% 차이")
