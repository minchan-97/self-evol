"""
learning_guide.py — 탐지 기반 학습 가이드 (Detection-Driven Learning)

핵심 발상: 탐지 기능을 '이상 잡기'가 아니라 '학습 나침반'으로 뒤집어 쓴다.

들어온 정보 각각에 대해 세 가지를 탐지하고, 그 신호로 학습 여부/우선순위를 정한다:
  ① known      : 이미 아는 것인가 (중복/기존 코퍼스와 유사) → 건너뛰기
  ② foreign    : 도메인 밖인가 (가드레일이 거부) → 오염 위험, 거부
  ③ novelty    : 빈 영역을 채우는 새것인가 → 우선 학습

이로써 "자기 정체성 안에서, 모르는 것만, 효율적으로" 자란다.
= 며칠간의 질문 "오염 없이 자기를 유지하며 자라는가"에 대한 구조적 답.
"""
from __future__ import annotations
import numpy as np
from selfloop_engine import tokenize


class LearningGuide:
    def __init__(self, state, emb,
                 known_sim_thr=0.85,      # 이보다 유사하면 '아는 것'
                 novelty_min=0.15):       # 이보다 새로워야 '학습 가치'
        self.state = state
        self.emb = emb
        self.known_sim_thr = known_sim_thr
        self.novelty_min = novelty_min
        self._corpus_cache = None
        self._corpus_n = 0
        self._corpus_set = set()

    def _corpus_matrix(self):
        """코퍼스 임베딩 캐시 (정규화)."""
        corpus = self.state.corpus
        if self._corpus_cache is None or self._corpus_n != len(corpus):
            if not corpus:
                self._corpus_cache = None
            else:
                M = self.emb.encode_many(corpus)
                self._corpus_cache = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
            self._corpus_n = len(corpus)
            self._corpus_set = set(corpus)
        return self._corpus_cache

    # ── 탐지 ① 아는 것인가 ──────────────────────────────
    def detect_known(self, sentence):
        """기존 코퍼스와 최대 유사도. 높거나 완전일치면 '이미 아는 것'."""
        M = self._corpus_matrix()
        # 완전 일치 — 가장 확실한 '아는 것'
        if sentence in self._corpus_set:
            return 1.0, True
        if M is None:
            return 0.0, False
        v = self.emb.encode(sentence)
        v = v / (np.linalg.norm(v) + 1e-12)
        sims = M @ v
        max_sim = float(np.max(sims))
        near_dup = max_sim >= self.known_sim_thr
        return max_sim, near_dup

    # ── 탐지 ② 도메인 밖인가 ────────────────────────────
    def detect_foreign(self, sentence):
        """가드레일 점수. 임계값 미만이면 도메인 밖(오염 위험)."""
        g = self.state.guardrail
        if g is None or not g.trained:
            return False, 0.0
        ok, score = g.judge(sentence, self.state.guard_threshold)
        return (not ok), score

    # ── 탐지 ③ 새 영역을 채우나 ─────────────────────────
    def detect_novelty(self, sentence):
        """
        SOM 관점의 새로움: 가장 가까운 노드와의 거리.
        멀수록(=잘 안 맞는 곳) 새 영역 → 학습 가치 높음.
        반환: 0~1 정규화 새로움 점수.
        """
        g = self.state.gsom
        if g.W.size == 0:
            return 1.0
        v = self.emb.encode(sentence)
        d = np.linalg.norm(g.W - v, axis=1)
        nearest = float(np.min(d))
        # 코퍼스 평균 노드간 거리로 정규화
        scale = float(np.mean(d)) + 1e-9
        return float(min(nearest / scale, 1.0))

    # ── 통합 판단 ───────────────────────────────────────
    def guide(self, sentence):
        """
        한 문장에 대한 학습 가이드 결정.
        반환: dict(action, priority, reasons)
          action: 'learn' | 'skip' | 'reject'
          priority: 학습 우선순위 (높을수록 먼저, learn일 때만 의미)
        """
        known, near_dup = self.detect_known(sentence)
        is_foreign, fscore = self.detect_foreign(sentence)
        novelty = self.detect_novelty(sentence)

        reasons = []
        # ② 도메인 밖 → 거부 (정체성 보호 최우선)
        if is_foreign:
            return {"action": "reject", "priority": 0.0,
                    "reasons": [f"도메인 밖(가드레일 {fscore:.1f})"],
                    "known": round(known, 2), "novelty": round(novelty, 2)}
        # ① 이미 아는 것(거의 동일) → 건너뛰기
        if near_dup:
            return {"action": "skip", "priority": 0.0,
                    "reasons": [f"이미 아는 것(유사도 {known:.2f})"],
                    "known": round(known, 2), "novelty": round(novelty, 2)}
        # ③ 너무 안 새로우면 학습 가치 낮음
        if novelty < self.novelty_min:
            return {"action": "skip", "priority": 0.0,
                    "reasons": [f"새로움 부족({novelty:.2f})"],
                    "known": round(known, 2), "novelty": round(novelty, 2)}
        # 학습 대상: 우선순위 = 새로움↑ + 덜아는것↑ (도메인 안은 보장됨)
        priority = 0.6 * novelty + 0.4 * (1 - known)
        reasons.append(f"새 영역(새로움 {novelty:.2f}, 미지 {1-known:.2f})")
        return {"action": "learn", "priority": round(float(priority), 3),
                "reasons": reasons,
                "known": round(known, 2), "novelty": round(novelty, 2)}

    def guide_batch(self, sentences):
        """여러 문장을 가이드하고, 학습 대상을 우선순위 정렬해 반환."""
        results = [(s, self.guide(s)) for s in sentences]
        learn = sorted([(s, g) for s, g in results if g["action"] == "learn"],
                       key=lambda x: x[1]["priority"], reverse=True)
        skip = [(s, g) for s, g in results if g["action"] == "skip"]
        reject = [(s, g) for s, g in results if g["action"] == "reject"]
        return {"learn": learn, "skip": skip, "reject": reject,
                "summary": {"learn": len(learn), "skip": len(skip),
                            "reject": len(reject)}}
