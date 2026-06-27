"""
build_tok_emb.py
================
코퍼스에서 word-level 임베딩(skip-gram + negative sampling)을 CPU로 학습해
앱이 읽는 tok_emb pkl을 생성한다.

출력 형식 (EmbeddingProvider가 인식):
    {"word2idx": {...}, "idx2word": {...}, "tok_emb": ndarray(V, dim), "dim": dim}

사용:
    python build_tok_emb.py --pkl 시계_임용.pkl --out tok_emb.pkl --dim 64
    python build_tok_emb.py --txt corpus.txt --out tok_emb.pkl
    # 상태 pkl에서 코퍼스를 꺼내거나, txt를 직접 주거나 둘 중 하나
"""

from __future__ import annotations
import argparse
import pickle
import re
import numpy as np
from collections import Counter


# ----------------------------------------------------------------------
# 토크나이즈: 한국어 어절 + 영문 단어. 너무 긴 토큰/기호는 정리.
# ----------------------------------------------------------------------
def tokenize(sentence: str):
    s = sentence.strip()
    # URL/이메일 제거
    s = re.sub(r"https?://\S+|\S+@\S+", " ", s)
    # 한글/영문/숫자만 남기고 분리
    toks = re.findall(r"[가-힣]+|[A-Za-z]+|[0-9]+", s)
    # 1글자 영문/숫자 노이즈 약간 정리(한글 1글자는 의미 있을 수 있어 유지)
    out = []
    for t in toks:
        if re.fullmatch(r"[A-Za-z0-9]", t):
            continue
        out.append(t.lower() if re.fullmatch(r"[A-Za-z]+", t) else t)
    return out


def build_vocab(sentences, min_count=2, max_vocab=20000):
    cnt = Counter()
    for s in sentences:
        cnt.update(tokenize(s))
    # 빈도 필터
    items = [(w, c) for w, c in cnt.items() if c >= min_count]
    items.sort(key=lambda x: -x[1])
    items = items[:max_vocab]
    word2idx = {w: i for i, (w, _) in enumerate(items)}
    idx2word = {i: w for w, i in word2idx.items()}
    freq = np.array([c for _, c in items], dtype=np.float64)
    return word2idx, idx2word, freq


# ----------------------------------------------------------------------
# skip-gram + negative sampling (순수 numpy, CPU)
# ----------------------------------------------------------------------
def train_skipgram(sentences, word2idx, freq, dim=64, window=2,
                   neg=5, epochs=5, lr=0.025, seed=0):
    rng = np.random.default_rng(seed)
    V = len(word2idx)
    # 중심/문맥 임베딩
    W_in = (rng.random((V, dim)) - 0.5) / dim
    W_out = np.zeros((V, dim))

    # negative sampling 분포 (unigram^0.75)
    p_neg = freq ** 0.75
    p_neg /= p_neg.sum()

    # subsampling 확률 (빈출어 다운샘플)
    t = 1e-3
    f = freq / freq.sum()
    p_keep = np.minimum(1.0, (np.sqrt(f / t) + 1) * (t / f))

    # 문장 → 인덱스 시퀀스
    corpus_idx = []
    for s in sentences:
        seq = [word2idx[w] for w in tokenize(s) if w in word2idx]
        if seq:
            corpus_idx.append(seq)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    n_pairs = 0
    for ep in range(epochs):
        cur_lr = lr * (1 - ep / max(1, epochs)) + lr * 0.1
        rng.shuffle(corpus_idx)
        loss_acc, cnt = 0.0, 0
        for seq in corpus_idx:
            for i, center in enumerate(seq):
                # subsampling
                if rng.random() > p_keep[center]:
                    continue
                win = rng.integers(1, window + 1)
                lo, hi = max(0, i - win), min(len(seq), i + win + 1)
                for j in range(lo, hi):
                    if j == i:
                        continue
                    ctx = seq[j]
                    # positive + negatives
                    negs = rng.choice(len(p_neg), size=neg, p=p_neg)
                    targets = np.concatenate(([ctx], negs))
                    labels = np.zeros(neg + 1); labels[0] = 1.0

                    v_in = W_in[center]
                    v_out = W_out[targets]              # (neg+1, dim)
                    score = sigmoid(v_out @ v_in)       # (neg+1,)
                    g = (score - labels)                # (neg+1,)
                    # 업데이트
                    grad_in = g @ v_out                 # (dim,)
                    W_out[targets] -= cur_lr * np.outer(g, v_in)
                    W_in[center] -= cur_lr * grad_in
                    loss_acc += -np.log(score[0] + 1e-10) - np.sum(np.log(1 - score[1:] + 1e-10))
                    cnt += 1
                    n_pairs += 1
        print(f"  epoch {ep+1}/{epochs}  lr={cur_lr:.4f}  pairs={cnt}  "
              f"avg_loss={loss_acc/max(1,cnt):.4f}")
    print(f"총 학습 페어: {n_pairs}")
    # 최종 임베딩 = 입력 임베딩 (정규화)
    norms = np.linalg.norm(W_in, axis=1, keepdims=True) + 1e-8
    return W_in / norms


def extract_corpus(args):
    if args.pkl:
        b = pickle.load(open(args.pkl, "rb"))
        if isinstance(b, dict) and "corpus" in b:
            return b["corpus"]
        raise SystemExit("pkl에 corpus 키가 없습니다.")
    if args.txt:
        with open(args.txt, encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    raise SystemExit("--pkl 또는 --txt 중 하나를 주세요.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", help="상태 pkl (corpus 추출)")
    ap.add_argument("--txt", help="줄단위 코퍼스 txt")
    ap.add_argument("--out", default="tok_emb.pkl")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--seed-emb", help="기존 tok_emb.pkl을 씨앗으로 불러와 이어학습(성장)")
    args = ap.parse_args()

    sents = extract_corpus(args)
    print(f"코퍼스 {len(sents)}문장 로드")

    # ---- 씨앗 성장 모드 ----
    if args.seed_emb:
        from selfloop_engine import GrowingEmbedding
        ge = GrowingEmbedding(dim=args.dim)
        ok, msg = ge.load_seed(tok_emb_path=args.seed_emb)
        print("씨앗:", msg)
        if not ok:
            raise SystemExit("씨앗 로드 실패")
        before = ge.vocab_size
        r = ge.grow(sents, epochs=args.epochs)
        print(f"성장 완료: 새 단어 {r['new_words']}개 (vocab {before}→{r['vocab']})")
        ge.save(args.out)
        print(f"저장: {args.out}")
        return

    # ---- 처음부터 학습 ----
    word2idx, idx2word, freq = build_vocab(sents, min_count=args.min_count)
    print(f"어휘 {len(word2idx)}개 (min_count={args.min_count})")
    print("skip-gram 학습 시작...")
    emb = train_skipgram(sents, word2idx, freq, dim=args.dim, window=5,
                         neg=8, epochs=args.epochs, lr=0.05)
    out = {"word2idx": word2idx, "idx2word": idx2word,
           "tok_emb": emb.astype(np.float32), "dim": args.dim}
    pickle.dump(out, open(args.out, "wb"))
    print(f"저장: {args.out}  (vocab={len(word2idx)}, dim={args.dim})")

    print("\n[품질 점검] 최근접 이웃:")
    def neighbors(word, k=5):
        if word not in word2idx:
            return f"'{word}' 어휘에 없음"
        v = emb[word2idx[word]]
        sims = emb @ v
        idx = np.argsort(-sims)[1:k+1]
        return ", ".join(f"{idx2word[i]}({sims[i]:.2f})" for i in idx)
    for w in ["시계", "워치", "임용", "교사", "교육", "다이아몬드"]:
        print(f"  {w}: {neighbors(w)}")


if __name__ == "__main__":
    main()
