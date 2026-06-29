"""
auto_learn.py — 자동 백그라운드 학습 엔진.

선생님 요구:
- 대화 후, 그 대화를 바탕으로 스스로 검색해서 계속 학습
- 다양한 분야를 자율적으로 넓혀가며 학습
- Brave 검색으로 질문 검색도 가능

동작:
  1) 씨앗 주제(대화/질문에서 추출 or SOM 빈 격자에서 자동 생성)
  2) Brave/폴백 검색으로 자료 수집
  3) 가드레일 필터 → 학습
  4) 보상 계산 → QueryPolicy 갱신 (좋은 방향 강화)
  5) 다음 주제 선택 (explore/exploit) → 반복

저사양/모바일 보호: 한 사이클씩 호출형(generator)으로 만들어
앱이 사이클마다 UI 갱신하고 멈출 수 있게.
"""
from __future__ import annotations
import time
import numpy as np
from selfloop_engine import (autonomous_queries, crawl_topic, measure,
                             learning_reward, tokenize)


# 자율 탐색에서 무관 분야로 새게 만드는 '너무 일반적인' 검색어들.
# 이런 단어로 검색하면 세상 모든 분야가 딸려와 정체성이 오염된다.
_TOO_GENERIC = {
    "실험", "관점", "방법", "결과", "내용", "경우", "문제", "사용", "활용",
    "분석", "처리", "기반", "통해", "대한", "위한", "이런", "그런", "여러",
    "다양", "관련", "정보", "자료", "시스템", "연구", "설계", "구조", "방식",
    "self", "organizing", "outlier", "smoothing", "data", "the", "and",
}


def _is_too_generic(query: str) -> bool:
    """검색어가 너무 일반적이면(무관 분야 유입 위험) True."""
    toks = [t for t in tokenize(query) if len(t) >= 2]
    if not toks:
        return True
    generic_hits = sum(1 for t in toks if t.lower() in _TOO_GENERIC)
    # 절반 이상이 일반어이거나, 단어가 1개뿐인데 일반어면 위험
    return generic_hits >= max(1, len(toks) // 2 + 1) or \
        (len(toks) == 1 and toks[0].lower() in _TOO_GENERIC)


def seed_topics_from_text(text: str, max_topics=3):
    """대화/질문 텍스트에서 명사 위주 씨앗 검색어 추출(조사·짧은 토큰 제거)."""
    toks = [t for t in tokenize(text) if len(t) >= 2]
    # 빈도 높은 순 간단 추출
    from collections import Counter
    cnt = Counter(toks)
    common = [w for w, _ in cnt.most_common(max_topics * 2)]
    # 2개씩 묶어 검색어로
    topics = []
    for i in range(0, min(len(common), max_topics * 2), 2):
        topics.append(" ".join(common[i:i + 2]))
    return topics[:max_topics] or (common[:max_topics] if common else [])


class AutoLearner:
    """한 사이클씩 실행하는 자율 학습기."""

    def __init__(self, state, emb, brave_api_key=None,
                 max_pages=4, train_rounds=3, use_guardrail=True):
        self.state = state
        self.emb = emb
        self.brave_api_key = brave_api_key
        self.max_pages = max_pages
        self.train_rounds = train_rounds
        self.use_guardrail = use_guardrail
        self.cycle = 0
        self.history = []  # 사이클별 결과 로그

    def _pick_query(self, seed_text=None):
        """다음 검색어 선택: 씨앗 텍스트 우선, 없으면 SOM 빈격자 자동생성, 정책으로 정렬."""
        cands = []
        if seed_text:
            cands += seed_topics_from_text(seed_text, max_topics=3)
        # SOM 기반 자동 검색어(다양성 확장)
        try:
            Xa = self.emb.encode_many(self.state.corpus) if self.state.corpus else None
            if Xa is not None and len(self.state.corpus) >= 5:
                cands += autonomous_queries(self.state.gsom, Xa, self.state.corpus,
                                            self.emb, n_queries=3)
        except Exception:
            pass
        cands = [c for c in cands if c and c.strip()]
        # 너무 일반적인 검색어 제거 (무관 분야 유입 차단)
        filtered = [c for c in cands if not _is_too_generic(c)]
        cands = filtered or cands  # 다 걸러지면 원본 유지(빈 검색 방지)
        if not cands:
            return None, "none"
        # 정책으로 정렬(explore/exploit)
        import random
        if hasattr(self.state, "policy") and self.state.policy is not None:
            ranked, mode = self.state.policy.rank_queries(cands, rng=random.Random())
            return ranked[0], mode
        return random.choice(cands), "random"

    def step(self, seed_text=None):
        """한 사이클: 검색어 선택 → 수집 → 학습 → 보상 → 정책 갱신."""
        self.cycle += 1
        q, mode = self._pick_query(seed_text)
        rec = {"cycle": self.cycle, "query": q, "mode": mode,
               "added": 0, "rejected": 0, "reward": None, "note": ""}
        if not q:
            rec["note"] = "검색어 생성 실패"
            self.history.append(rec)
            return rec

        # 수집
        try:
            sents, _log, _links = crawl_topic(
                q, max_pages=self.max_pages, return_sources=True,
                brave_api_key=self.brave_api_key)
        except Exception as e:
            rec["note"] = f"수집 실패: {e}"
            if hasattr(self.state, "policy"):
                self.state.policy.record(q, -0.1)
            self.history.append(rec)
            return rec

        if not sents:
            rec["note"] = "0 문장 수집"
            if hasattr(self.state, "policy"):
                self.state.policy.record(q, -0.1)
            self.history.append(rec)
            return rec

        # 가드레일 필터 + 탐지 기반 학습 가이드로 추가
        # 0차: 정체성 기준선(오염 안 된 원래 기준)으로 외래 선차단
        # 1차: 가드레일 → 2차: 학습가이드(아는것 skip, 새것 우선)
        if getattr(self.state, "baseline_locked", False):
            sents = [s for s in sents if self.state.check_identity(s)[0]]
            if not sents:
                rec["note"] = "정체성 기준선이 전부 외래로 차단"
                if hasattr(self.state, "policy"):
                    self.state.policy.record(q, -0.2)
                self.history.append(rec)
                return rec
        try:
            from learning_guide import LearningGuide
            lg = LearningGuide(self.state, self.emb)
            batch = lg.guide_batch(sents)
            learn_sents = [s for s, _ in batch["learn"]]
            rec["skipped"] = batch["summary"]["skip"]
            rec["guide_rejected"] = batch["summary"]["reject"]
            if learn_sents:
                added, rejected = self.state.add_sentences(
                    learn_sents, use_guardrail=self.use_guardrail)
            else:
                added, rejected = 0, len(sents)
        except Exception:
            added, rejected = self.state.add_sentences(
                sents, use_guardrail=self.use_guardrail)
        rec["added"], rec["rejected"] = added, rejected
        if added == 0:
            rec["note"] = "전부 도메인 밖 거부"
            if hasattr(self.state, "policy"):
                self.state.policy.record(q, -0.2)
            self.history.append(rec)
            return rec

        # 학습 전후 측정 + 보상
        try:
            X = self.emb.encode_many(self.state.corpus)
            toks = [s.split() for s in self.state.corpus]
            m0 = measure(self.state.gsom, X, toks)
            for _ in range(self.train_rounds):
                self.state.gsom.round += 1
                lr = max(0.02, 0.4 * (0.9 ** self.state.gsom.round))
                rad = max(0.5, 2.0 * (0.85 ** self.state.gsom.round))
                self.state.gsom.train_step(X, lr, rad)
                self.state.gsom.grow()
            m1 = measure(self.state.gsom, X, toks)
            m1["round"] = self.state.gsom.round
            self.state.history.append(m1)
            reward, detail = learning_reward(
                m0["mean_qe"], m1["mean_qe"], m0["vocab_div"], m1["vocab_div"])
            rec["reward"] = round(reward, 3)
            rec["detail"] = detail
            if hasattr(self.state, "policy"):
                self.state.policy.record(q, reward)
            # 가드레일 재적합(코퍼스 커졌으니)
            if len(self.state.corpus) >= 10:
                self.state.fit_guardrail(percentile=25)
            rec["note"] = "학습 완료"
        except Exception as e:
            rec["note"] = f"학습 오류: {e}"

        self.history.append(rec)
        return rec
