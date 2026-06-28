"""
experts.py — 멀티 전문가 구조 ("넓으면서 안 잡탕").

각 분야 = 독립된 SelfLoopState (SOM + 가드레일 + 코퍼스).
라우터가 질문을 받아 '어느 전문가의 도메인인가'를 가드레일 점수로 판별해
가장 적합한 전문가에게 보낸다.

장점:
- 각 전문가는 단일 도메인이라 깊게 수렴(깨끗)
- 전체적으로는 여러 분야를 커버(넓음)
- 새 분야는 새 전문가로 추가 → 기존 전문가 오염 없음

핵심: 며칠간의 결론("단일 도메인은 깊게, 잡탕은 바닥이 높다")을
구조로 해결 — 하나로 넓히지 말고 여러 개의 깊은 전문가로.
"""
from __future__ import annotations
import os
import numpy as np
from selfloop_engine import SelfLoopState, EmbeddingProvider, GrowingSOM, measure


class ExpertRouter:
    def __init__(self, dim=64, emb=None):
        self.dim = dim
        self.emb = emb or EmbeddingProvider(dim=dim)
        self.experts: dict[str, SelfLoopState] = {}   # 이름 -> 상태
        self.meta: dict[str, dict] = {}               # 이름 -> {desc, ...}

    # ---------------- 전문가 관리 ----------------
    def add_expert(self, name: str, desc: str = ""):
        if name in self.experts:
            return self.experts[name]
        st = SelfLoopState(dim=self.dim)
        self.experts[name] = st
        self.meta[name] = {"desc": desc}
        return st

    def remove_expert(self, name: str):
        self.experts.pop(name, None)
        self.meta.pop(name, None)

    def list_experts(self):
        return [{"name": n, "corpus": len(s.corpus), "nodes": s.gsom.n,
                 "desc": self.meta.get(n, {}).get("desc", "")}
                for n, s in self.experts.items()]

    # ---------------- 라우팅 ----------------
    def route(self, text: str, min_margin=0.5, abs_floor=-5.0):
        """
        각 전문가 가드레일에게 '이 텍스트 네 도메인이냐'를 물어 점수 수집.
        - 1등 점수가 abs_floor 미만이면 → 아무도 자신 없음 → None(새 분야 후보)
        - 1등과 2등 차이가 min_margin 미만이면 → 애매함 → None
        반환: (best_name, scores_dict, info)
        """
        scores = {}
        for name, st in self.experts.items():
            if st.guardrail is None or not st.guardrail.trained:
                scores[name] = None
                continue
            ok, sc = st.guardrail.judge(text, st.guard_threshold)
            scores[name] = {"ok": ok, "score": round(sc, 2)}

        ranked = sorted(
            [(n, v["score"]) for n, v in scores.items() if v],
            key=lambda x: x[1], reverse=True)
        if not ranked:
            return None, scores, {"reason": "전문가 없음"}

        top_name, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else -99.0
        margin = top_score - second_score

        if top_score < abs_floor:
            return None, scores, {"reason": "모두 자신 없음(새 분야 후보)",
                                  "top": top_name, "top_score": top_score}
        if margin < min_margin:
            return None, scores, {"reason": "애매함(여러 분야 경합)",
                                  "top": top_name, "margin": round(margin, 2)}
        return top_name, scores, {"reason": "라우팅 성공",
                                  "top_score": top_score, "margin": round(margin, 2)}

    # ---------------- 학습 ----------------
    def train_expert(self, name: str, rounds=40):
        """특정 전문가의 SOM 학습 + 가드레일 적합."""
        st = self.experts[name]
        if not st.corpus:
            return {"error": "코퍼스 없음"}
        X = self.emb.encode_many(st.corpus)
        emb_dim = X.shape[1]
        som_dim = st.gsom.W.shape[1] if st.gsom.W.size else emb_dim
        if emb_dim != som_dim:
            st.gsom = GrowingSOM(dim=emb_dim, init_nodes=36, seed=0)
            rng = np.random.default_rng(0)
            idx = rng.choice(len(X), min(36, len(X)), replace=False)
            st.gsom.W = X[idx].copy() + rng.normal(scale=0.01, size=(len(idx), emb_dim))
            st.gsom.coords = st.gsom.coords[:len(idx)]
            st.gsom.err = st.gsom.err[:len(idx)]
        toks = [s.split() for s in st.corpus]
        for r in range(rounds):
            st.gsom.round += 1
            lr = max(0.02, 0.4 * (0.9 ** st.gsom.round))
            rad = max(0.5, 2.0 * (0.85 ** st.gsom.round))
            st.gsom.train_step(X, lr, rad)
            st.gsom.grow()
        if len(st.corpus) >= 10:
            st.fit_guardrail(percentile=25)
        m = measure(st.gsom, X, toks)
        return {"qe": round(m["mean_qe"], 2), "vocab": round(m["vocab_div"], 3),
                "nodes": st.gsom.n, "threshold": round(st.guard_threshold, 2)}

    def add_to_expert(self, name: str, sentences, use_guardrail=False):
        st = self.experts[name]
        return st.add_sentences(sentences, use_guardrail=use_guardrail)

    # ---------------- 저장/복원 ----------------
    def save(self, path: str):
        import pickle
        blob = {"dim": self.dim, "experts": {}, "meta": self.meta}
        for name, st in self.experts.items():
            # SelfLoopState.save 의 직렬화 재사용 위해 임시 dict 구성
            from selfloop_engine import Quantizer
            qz = Quantizer(st.gsom.W) if st.gsom.W.size else None
            blob["experts"][name] = {
                "corpus": st.corpus,
                "W_int8": qz.q(st.gsom.W) if qz else None,
                "W_scale": qz.scale if qz else None,
                "coords": st.gsom.coords, "err": st.gsom.err,
                "round": st.gsom.round, "history": st.history,
                "guard_threshold": st.guard_threshold,
            }
        with open(path, "wb") as f:
            pickle.dump(blob, f)

    @classmethod
    def load(cls, path: str, emb=None):
        import pickle
        from selfloop_engine import MarkovGuardrail
        with open(path, "rb") as f:
            blob = pickle.load(f)
        r = cls(dim=blob["dim"], emb=emb)
        r.meta = blob.get("meta", {})
        for name, e in blob["experts"].items():
            st = SelfLoopState(dim=blob["dim"])
            st.corpus = e["corpus"]
            if e["W_int8"] is not None:
                st.gsom.W = e["W_int8"].astype(np.float64) * e["W_scale"]
                st.gsom.coords = e["coords"]; st.gsom.err = e["err"]
                st.gsom.round = e["round"]
            st.history = e.get("history", [])
            st.guard_threshold = e.get("guard_threshold", -9.5)
            # 가드레일 재구성
            if st.corpus:
                st.guardrail = MarkovGuardrail().fit(st.corpus)
            r.experts[name] = st
        return r
