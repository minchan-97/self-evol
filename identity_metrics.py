"""
identity_metrics.py — 정체성 유지 지표 (Identity Retention)

핵심 질문: "스스로 학습하면서 자기 자신을 유지하는가?"

자율 학습은 코퍼스에 새 문장을 계속 쌓는다. 문제는 그 유입이
초기 정체성(핵심 도메인)에서 얼마나 벗어나느냐다.

측정 (코퍼스를 시간순으로 본다는 가정 — append 순서 = 학습 순서):
1. seed_vocab_retention  : 초기 핵심 어휘가 전체에서 차지하는 비중 유지율
2. drift_distance        : 초기 중심 vs 나중 유입의 의미 거리 (멀수록 오염)
3. coherence             : 전체 코퍼스의 내부 응집도 (낮을수록 잡탕)
4. foreign_ratio         : 초기 도메인 가드레일이 '밖'으로 판정하는 나중 문장 비율

종합 점수 IR(0~1): 높을수록 '자기를 지키며 자람'.
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from selfloop_engine import tokenize


def _vocab_set(sentences, top=None):
    cnt = Counter()
    for s in sentences:
        cnt.update(tokenize(s))
    if top:
        return set(w for w, _ in cnt.most_common(top))
    return set(cnt)


def seed_vocab_retention(corpus, seed_frac=0.3, top=100):
    """초기 seed_frac 구간의 핵심 어휘가 후반부에 얼마나 살아있나."""
    n = len(corpus)
    if n < 10:
        return 1.0
    k = max(5, int(n * seed_frac))
    seed = corpus[:k]
    later = corpus[k:]
    seed_vocab = _vocab_set(seed, top=top)
    if not seed_vocab:
        return 1.0
    later_vocab = _vocab_set(later)
    keep = len(seed_vocab & later_vocab) / len(seed_vocab)
    return float(keep)


def drift_distance(corpus, emb, seed_frac=0.3):
    """초기 중심 벡터 vs 후반 유입 중심 벡터의 의미 거리(코사인 거리)."""
    n = len(corpus)
    if n < 10:
        return 0.0
    k = max(5, int(n * seed_frac))
    seed = corpus[:k]
    later = corpus[k:]
    if not later:
        return 0.0
    sv = emb.encode_many(seed).mean(axis=0)
    lv = emb.encode_many(later).mean(axis=0)
    sv = sv / (np.linalg.norm(sv) + 1e-12)
    lv = lv / (np.linalg.norm(lv) + 1e-12)
    cos = float(np.dot(sv, lv))
    return (1.0 - cos) / 2.0  # 0(동일)~1(정반대)


def coherence(corpus, emb, sample=200):
    """전체 코퍼스 내부 응집도: 평균 문장이 전체 중심에 얼마나 가까운가."""
    n = len(corpus)
    if n < 5:
        return 1.0
    idx = np.random.default_rng(0).choice(n, min(sample, n), replace=False)
    M = emb.encode_many([corpus[i] for i in idx])
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    center = Mn.mean(axis=0)
    center /= (np.linalg.norm(center) + 1e-12)
    return float(np.mean(Mn @ center))  # 1에 가까울수록 응집


def foreign_ratio(corpus, seed_frac=0.3, percentile=25):
    """
    초기 도메인 가드레일이 후반 유입 문장을 '밖'으로 판정하는 비율.
    주의: 이 값의 해석은 맥락에 따라 다르다.
      - 이미 쌓인 코퍼스 측정 시: 높으면 '오염됨'(외래가 많이 섞임)
      - 방어 검증 시: 가드레일이 외래를 잘 거른다는 뜻(방어율)
    그래서 identity_retention 에서는 이 값을 종합점수에 직접 넣지 않고
    참고 지표로만 보고한다.
    """
    from selfloop_engine import MarkovGuardrail
    n = len(corpus)
    if n < 20:
        return 0.0
    k = max(10, int(n * seed_frac))
    seed = corpus[:k]
    later = corpus[k:]
    g = MarkovGuardrail().fit(seed)
    thr = g.suggest_threshold(seed, percentile=percentile)
    rejected = sum(1 for s in later if not g.judge(s, thr)[0])
    return float(rejected / max(len(later), 1))


def identity_retention(corpus, emb, seed_frac=0.3):
    """
    종합 정체성 유지 점수 (0~1, 높을수록 자기 유지).

    종합점수는 '코퍼스 자체가 얼마나 자기다운가'만 본다:
      - seed_vocab_retention (초기 어휘 보존)
      - drift_distance       (의미 이동)
      - coherence            (내부 응집)
    foreign_ratio 는 해석이 양날이라 종합에서 빼고 참고로만 보고.
    """
    svr = seed_vocab_retention(corpus, seed_frac)
    dd = drift_distance(corpus, emb, seed_frac)
    coh = coherence(corpus, emb)
    fr = foreign_ratio(corpus, seed_frac)
    # 종합: 어휘유지↑, 응집↑ 좋고 / 드리프트↑ 나쁨 (외래비율 제외)
    ir = 0.40 * svr + 0.30 * (1 - dd) + 0.30 * coh
    return {
        "IR": round(float(ir), 3),
        "seed_vocab_retention": round(svr, 3),
        "drift_distance": round(dd, 3),
        "coherence": round(coh, 3),
        "foreign_ratio_ref": round(fr, 3),  # 참고용(종합 제외)
    }


def guardrail_defense_rate(clean_seed, foreign_samples, percentile=25):
    """
    방어 검증용 별도 지표: 깨끗한 seed 가드레일이
    외래 샘플을 얼마나 차단하는가 (높을수록 방어 강함).
    """
    from selfloop_engine import MarkovGuardrail
    if not clean_seed or not foreign_samples:
        return 0.0
    g = MarkovGuardrail().fit(clean_seed)
    thr = g.suggest_threshold(clean_seed, percentile=percentile)
    blocked = sum(1 for s in foreign_samples if not g.judge(s, thr)[0])
    return float(blocked / len(foreign_samples))
