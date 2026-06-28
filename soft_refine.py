"""
soft_refine.py — 선생님 제안 구현:
  "관심 영역만 역양자화 → 소프트맥스 할당 → 오차 기반 국소 미세조정"

핵심 아이디어:
- 평소 SOM 가중치는 INT8 양자화로 보관(가벼움).
- 질문/관심 입력이 들어오면, 그 입력 근처 노드(top-k)만 FP로 역양자화.
- 그 국소 영역에서 소프트맥스로 '부드러운 할당' 계산.
- 입력을 목표로 두고 '오차(입력-노드)'를 소프트맥스 가중치만큼 흘려
  그 영역 노드만 미세조정 → 빠진 조각(오차 기반 학습) 추가.

이것이 "유사 신경망"인가에 대한 정직한 답:
- 역전파(다층 미분 연쇄)는 아니다. 단층 국소 업데이트다.
- 그러나 '오차를 줄이는 방향으로 가중치를 미는' 점에서 신경망의 정신은 닿는다.
- 효과는 숫자(국소 QE 감소)로 측정한다.
"""
from __future__ import annotations
import numpy as np


def softmax(z, temp=1.0):
    z = np.asarray(z, dtype=np.float64) / max(temp, 1e-6)
    z -= z.max()
    e = np.exp(z)
    return e / (e.sum() + 1e-12)


class SoftRefiner:
    """
    W_q: INT8 양자화 가중치 (n, dim)
    scale: 역양자화 스케일 (dim,)  -- Quantizer.scale 과 동일
    """
    def __init__(self, W_q, scale, temp=0.5, lr=0.2):
        self.W_q = W_q.astype(np.int8)
        self.scale = np.asarray(scale, dtype=np.float64)
        self.temp = temp           # 소프트맥스 온도(작을수록 BMU에 가깝게 날카로움)
        self.lr = lr

    def _dq_rows(self, idx):
        """관심 노드(idx)만 역양자화."""
        return self.W_q[idx].astype(np.float64) * self.scale

    def soft_assign(self, x, k=8):
        """
        입력 x 근처 top-k 노드만 역양자화 → 소프트맥스 할당 확률 반환.
        반환: (idx, probs, W_local)

        최적화: 후보 추림은 INT8 상태에서 직접 거리 계산(역양자화 없이).
        x 도 같은 scale 로 양자화해서 정수공간에서 근사 거리 → top-k 만 FP 역양자화.
        """
        # x 를 int 공간으로 (scale 로 나눠 정수 근사)
        xq = x / self.scale  # (dim,) float, 노드 int8 과 같은 척도
        # INT8 노드와의 거리 제곱을 양자화 공간에서 근사 계산 (역양자화 X)
        Wq = self.W_q.astype(np.float64)
        diff = Wq - xq
        d2 = np.einsum("nd,nd->n", diff, diff)  # 양자화 공간 근사 거리
        idx = np.argpartition(d2, min(k, len(d2) - 1))[:k]
        # 이 k개만 진짜 역양자화 (관심 영역만 정밀)
        W_local = self._dq_rows(idx)
        d2_local = np.einsum("nd,nd->n", W_local - x, W_local - x)
        probs = softmax(-d2_local, temp=self.temp)
        return idx, probs, W_local

    def local_qe(self, x, k=8):
        """국소 양자화오차(soft): 소프트맥스 가중 평균 거리."""
        idx, probs, W_local = self.soft_assign(x, k)
        dists = np.linalg.norm(W_local - x, axis=1)
        return float((probs * dists).sum())

    def refine(self, x, k=8):
        """
        오차 기반 국소 미세조정:
        관심 노드들을 입력 x 쪽으로 소프트맥스 가중치만큼 당김.
        업데이트 후 다시 양자화해서 저장(경량 유지).
        반환: (업데이트 전 국소QE, 업데이트 후 국소QE)
        """
        idx, probs, W_local = self.soft_assign(x, k)
        qe_before = float((probs * np.linalg.norm(W_local - x, axis=1)).sum())

        # 오차 = (x - 노드). 소프트맥스 확률 * lr 만큼 이동
        delta = (self.lr * probs)[:, None] * (x - W_local)
        W_new = W_local + delta

        # 다시 양자화해서 저장(해당 노드만)
        self.W_q[idx] = np.clip(np.round(W_new / self.scale), -127, 127).astype(np.int8)

        # 사용 빈도 기록(망각용)
        if not hasattr(self, "use_count"):
            self.use_count = np.zeros(self.W_q.shape[0])
        self.use_count[idx] += probs  # 많이 쓰인 노드일수록 누적

        # 업데이트 후 국소QE
        W_local2 = self._dq_rows(idx)
        d2 = np.einsum("nd,nd->n", W_local2 - x, W_local2 - x)
        probs2 = softmax(-d2, temp=self.temp)
        qe_after = float((probs2 * np.sqrt(d2)).sum())
        return qe_before, qe_after

    # ============================================================
    # 1. 자동 튜닝: 영역 크기/분산에 따라 temp, lr, k 자동 결정
    # ============================================================
    def auto_params(self, region_vecs):
        """관심 영역 입력들의 분산을 보고 temp/lr/k를 자동 설정."""
        R = np.asarray(region_vecs)
        m = R.shape[0]
        # 영역이 넓게 퍼져 있으면 k와 temp를 키워 부드럽게,
        # 좁으면 작게 해서 날카롭게.
        spread = float(np.mean(np.std(R, axis=0))) if m > 1 else 0.1
        k = int(np.clip(round(m * 2), 4, 16))
        temp = float(np.clip(spread * 8.0, 0.8, 4.0))
        lr = float(np.clip(0.4 - spread, 0.15, 0.35))
        self.temp, self.lr = temp, lr
        return {"k": k, "temp": round(temp, 2), "lr": round(lr, 2),
                "spread": round(spread, 3), "region_size": m}

    def refine_region(self, region_vecs, passes=8, k=None):
        """영역 단위 미세조정(과적합 방지). 자동 파라미터 사용."""
        p = self.auto_params(region_vecs)
        kk = k or p["k"]
        for _ in range(passes):
            for xv in region_vecs:
                self.refine(np.asarray(xv), k=kk)
        return p

    # ============================================================
    # 2. 관심 영역 자동 감지: 최근 입력들을 군집해 '영역'으로 묶기
    # ============================================================
    @staticmethod
    def detect_regions(recent_vecs, sim_threshold=0.6):
        """
        최근 대화/질문 벡터들을 코사인 유사도로 묶어 관심 영역(군집) 추출.
        간단한 그리디 군집(라이브러리 의존 없이).
        반환: [[vec,...], ...] 영역별 벡터 리스트
        """
        V = [np.asarray(v) / (np.linalg.norm(v) + 1e-8) for v in recent_vecs]
        used = [False] * len(V)
        regions = []
        for i in range(len(V)):
            if used[i]:
                continue
            group = [recent_vecs[i]]
            used[i] = True
            for j in range(i + 1, len(V)):
                if used[j]:
                    continue
                if float(V[i] @ V[j]) >= sim_threshold:
                    group.append(recent_vecs[j])
                    used[j] = True
            regions.append(group)
        return regions

    # ============================================================
    # 3. 망각: 오래 안 쓴 노드는 양자화를 거칠게(정보 흐리기)
    # ============================================================
    def forget(self, decay=0.95, coarse_factor=1.5):
        """
        use_count 가 낮은(안 쓰인) 노드의 가중치를 거칠게 만들어
        '흐릿한 기억'으로. 자주 쓰인 영역은 또렷하게 유지.
        """
        if not hasattr(self, "use_count"):
            return {"forgotten": 0}
        self.use_count *= decay  # 시간이 지나면 사용도 자체가 감쇠
        threshold = np.percentile(self.use_count, 30)  # 하위 30%
        cold = np.where(self.use_count <= threshold)[0]
        # 차가운 노드: 역양자화→거친 양자화(정밀도 의도적 손실)
        for i in cold:
            w = self.W_q[i].astype(np.float64) * self.scale
            coarse_scale = self.scale * coarse_factor
            self.W_q[i] = np.clip(np.round(w / coarse_scale) * coarse_factor,
                                  -127, 127).astype(np.int8)
        return {"forgotten": int(len(cold)), "kept_sharp": int(self.W_q.shape[0] - len(cold))}

