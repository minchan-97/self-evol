"""
app_selfloop.py
===============
자율 학습 루프 + 답변 루프 통합 Streamlit 앱.

실행:
    pip install streamlit numpy openai
    streamlit run app_selfloop.py

페이지
  1) 학습 루프  : 코퍼스 업로드 / "분야 검색 학습"(크롤링) / 라운드 옵션 / 계측 / pkl 저장
  2) 답변 루프  : pkl 불러오기 / 질문 / SOM 가드레일 / LLM 호출 / 추론경로

tok_emb 연결: 사이드바에서 tok_emb pkl 업로드 (없으면 해시 임베딩 폴백)
이전 학습 누적: pkl 불러오기 -> 같은 상태에 계속 학습
"""

import io
import os
import time
import pickle
import numpy as np
import streamlit as st

from selfloop_engine import (
    EmbeddingProvider, SelfLoopState, measure, collapse_warning,
    crawl_topic, llm_answer,
)

st.set_page_config(page_title="SelfLoop SOM", layout="wide",
                   initial_sidebar_state="expanded")

# ---------------------------------------------------------------- style
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
.stApp { background:#0d1117; color:#c9d1d9; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif; letter-spacing:-.02em; color:#e6edf3; }
.mono, code, pre { font-family:'JetBrains Mono',monospace !important; }
.metric-card{ background:#161b22; border:1px solid #21262d; border-left:3px solid #2dd4bf;
  padding:14px 16px; border-radius:6px; }
.metric-val{ font-family:'JetBrains Mono',monospace; font-size:1.7rem; color:#2dd4bf; font-weight:600;}
.metric-lab{ font-size:.72rem; text-transform:uppercase; letter-spacing:.08em; color:#8b949e;}
.warn{ background:#3d1b1b; border-left:3px solid #f85149; padding:10px 14px; border-radius:6px;
  color:#ffa198; font-family:'JetBrains Mono',monospace; font-size:.85rem;}
.ok{ background:#12261e; border-left:3px solid #2dd4bf; padding:10px 14px; border-radius:6px;
  color:#7ee2cf; font-family:'JetBrains Mono',monospace; font-size:.85rem;}
.stButton>button{ background:#2dd4bf; color:#0d1117; border:none; font-weight:600;
  font-family:'Space Grotesk',sans-serif; border-radius:6px;}
.stButton>button:hover{ background:#5eead4; color:#0d1117;}
.pathbox{ background:#161b22; border:1px solid #21262d; padding:12px; border-radius:6px;
  font-family:'JetBrains Mono',monospace; font-size:.8rem; color:#8b949e;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- state
if "state" not in st.session_state:
    st.session_state.state = SelfLoopState(dim=64)
if "emb" not in st.session_state:
    st.session_state.emb = EmbeddingProvider(dim=64)

def metric(col, label, value):
    col.markdown(f'<div class="metric-card"><div class="metric-val">{value}</div>'
                 f'<div class="metric-lab">{label}</div></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### ⚙ 설정")
    up = st.file_uploader("tok_emb pkl (선택)", key="tokemb",
                          help="TinyTransformer tok_emb가 담긴 .pkl")
    if up is not None and up.name.lower().endswith(".pkl"):
        tmp = f"/tmp/{up.name}"
        with open(tmp, "wb") as f:
            f.write(up.getbuffer())
        st.session_state.emb = EmbeddingProvider(dim=64, tok_emb_path=tmp)
        st.session_state.state.dim = st.session_state.emb.dim
    mode = st.session_state.emb.mode
    emb_obj = st.session_state.emb
    if mode == "tok_emb":
        st.markdown(f'<div class="ok">임베딩: TinyTransformer tok_emb<br>'
                    f'vocab={emb_obj.vocab_size} · dim={emb_obj.dim}</div>',
                    unsafe_allow_html=True)
    else:
        msg = "해시 폴백(tok_emb 없음)"
        if getattr(emb_obj, "load_error", None):
            msg = f"해시 폴백 · 로드 실패: {emb_obj.load_error}"
        st.markdown(f'<div class="warn">임베딩: {msg}</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🔌 GPT / LLM API")
    api_key_input = st.text_input(
        "API Key",
        value=os.environ.get("OPENAI_API_KEY", ""),
        type="password",
        help="OpenAI 또는 OpenAI 호환 서버의 API Key. 입력값은 세션에서만 사용됩니다."
    )
    base_url_input = st.text_input(
        "Base URL",
        value=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="OpenAI 기본값: https://api.openai.com/v1 / LM Studio 예: http://localhost:1234/v1"
    )
    max_tokens_input = st.number_input("max_tokens", min_value=128, max_value=4096, value=700, step=128)

    st.divider()
    stt = st.session_state.state
    st.markdown(f"**코퍼스** `{len(stt.corpus)}` 문장")
    st.markdown(f"**노드** `{stt.gsom.n}`")
    st.markdown(f"**라운드 누적** `{stt.gsom.round}`")

    st.divider()
    st.markdown("### 💾 세션")
    try:
        stt.save("/tmp/_save.pkl")
        with open("/tmp/_save.pkl", "rb") as f:
            st.download_button(
                "⬇ 현재 상태 저장 (selfloop_state.pkl)",
                f.read(),
                file_name="selfloop_state.pkl",
                help=f"코퍼스 {len(stt.corpus)}문장 · {stt.gsom.n}노드 · "
                     f"라운드 {stt.gsom.round}",
            )
    except Exception as e:
        st.markdown(f'<div class="warn">저장 준비 실패: {e}</div>',
                    unsafe_allow_html=True)
    rl = st.file_uploader("pkl 불러오기(이전 학습 누적)", key="loadpkl",
                          help="저장한 selfloop_state.pkl 파일을 선택하세요.")
    if rl is not None:
        st.caption(f"📄 파일 받음: {rl.name} ({rl.size:,} bytes)")
        if not rl.name.lower().endswith(".pkl"):
            st.markdown('<div class="warn">.pkl 파일만 불러올 수 있습니다. '
                        '(선택한 파일: ' + rl.name + ')</div>',
                        unsafe_allow_html=True)
        else:
            sig = (rl.name, rl.size)
            if st.session_state.get("_loaded_sig") != sig:
                try:
                    with open("/tmp/_load.pkl", "wb") as f:
                        f.write(rl.getbuffer())
                    loaded = SelfLoopState.load("/tmp/_load.pkl")
                    st.session_state.state = loaded
                    st.session_state._loaded_sig = sig
                    st.success(f"복원 완료: {len(loaded.corpus)}문장 · "
                               f"{loaded.gsom.n}노드 · dim={loaded.dim} · "
                               f"라운드 {loaded.gsom.round}")
                    if st.session_state.emb.dim != loaded.dim:
                        st.markdown(
                            f'<div class="warn">주의: 현재 임베딩 dim'
                            f'({st.session_state.emb.dim}) ≠ 불러온 상태 dim'
                            f'({loaded.dim}).<br>같은 임베딩으로 학습을 이어가려면 '
                            f'저장 당시와 동일한 tok_emb(또는 해시)로 맞추세요.</div>',
                            unsafe_allow_html=True)
                except Exception as e:
                    st.session_state._loaded_sig = None
                    st.markdown(f'<div class="warn">불러오기 실패: '
                                f'{type(e).__name__}: {e}</div>',
                                unsafe_allow_html=True)
            else:
                st.caption(f"이미 불러옴: {rl.name} "
                           f"({len(st.session_state.state.corpus)}문장)")
            if st.button("불러온 상태 초기화(빈 코퍼스로)"):
                st.session_state.state = SelfLoopState(dim=st.session_state.emb.dim)
                st.session_state._loaded_sig = None
                st.rerun()

# ---------------------------------------------------------------- header
st.markdown("# SelfLoop SOM")
st.markdown("<span class='mono' style='color:#8b949e'>자율 성장 의미지도 · 학습 루프와 답변 루프</span>",
            unsafe_allow_html=True)

page = st.tabs(["◐  학습 루프", "◑  답변 루프", "⚡ 정밀 학습", "♾️ 연속 자동학습"])

# ======================================================================
# PAGE 1 — 학습 루프
# ======================================================================
with page[0]:
    stt = st.session_state.state
    emb = st.session_state.emb

    st.markdown("### 1 · 학습 자료")
    src = st.radio("자료 출처",
                   ["코퍼스 직접 입력/업로드", "분야 검색해서 학습(크롤링)",
                    "🤖 자율 탐색 (빈 격자가 스스로 검색)"],
                   horizontal=True)

    new_sentences = []
    if src == "🤖 자율 탐색 (빈 격자가 스스로 검색)":
        st.markdown("#### 🤖 자율 탐색 학습")
        st.caption("GSOM의 저밀도 영역(덜 배운 곳)에서 스스로 검색어를 만들어 크롤링하고, "
                   "마르코프 가드레일을 통과한 문장만 학습합니다. 인간 지정 없이 자라납니다.")
        if len(stt.corpus) < 20:
            st.markdown('<div class="warn">자율 탐색은 먼저 어느 정도 학습된 코퍼스가 '
                        '필요합니다(최소 20문장). 코퍼스를 먼저 학습시키세요.</div>',
                        unsafe_allow_html=True)
        else:
            ac1, ac2, ac3 = st.columns(3)
            auto_brave = ac1.text_input("Brave API Key", type="password",
                                        value=os.environ.get("BRAVE_API_KEY", ""),
                                        key="auto_brave")
            auto_nq = ac2.slider("생성할 검색어 수", 1, 6, 3)
            auto_pages = ac3.slider("검색어당 페이지", 1, 5, 3)
            # 미리보기: 지금 생성될 검색어
            if st.button("🔮 자율 검색어 미리보기"):
                from selfloop_engine import autonomous_queries
                Xa = emb.encode_many(stt.corpus)
                qs = autonomous_queries(stt.gsom, Xa, stt.corpus, emb, n_queries=auto_nq)
                if qs:
                    st.markdown('<div class="ok">생성된 검색어: '
                                + " · ".join(f"<b>{q}</b>" for q in qs) + '</div>',
                                unsafe_allow_html=True)
                    st.session_state._auto_queries = qs
                else:
                    st.markdown('<div class="warn">검색어를 생성하지 못했습니다 '
                                '(코퍼스가 더 필요).</div>', unsafe_allow_html=True)
            auto_rounds = st.slider("검색어당 학습 라운드", 1, 10, 3, key="auto_rounds")
            if st.button("🤖 자율 수집 + 학습 (보상 루프)"):
                from selfloop_engine import (autonomous_queries, crawl_topic,
                                             measure, learning_reward)
                import numpy as _np
                # 1) 후보 검색어 생성 → 정책으로 정렬(explore/exploit)
                Xa = emb.encode_many(stt.corpus)
                cands = autonomous_queries(stt.gsom, Xa, stt.corpus, emb,
                                           n_queries=auto_nq * 2)
                if not cands:
                    st.markdown('<div class="warn">검색어 생성 실패.</div>',
                                unsafe_allow_html=True)
                else:
                    import random as _r
                    ranked, mode = stt.policy.rank_queries(cands, rng=_r.Random())
                    queries = ranked[:auto_nq]
                    st.caption(f"정책 모드: {mode} · 검색어: {', '.join(queries)}")
                    # 가드레일 준비
                    if len(stt.corpus) >= 10:
                        stt.fit_guardrail(percentile=25)
                    results = []
                    for q in queries:
                        with st.spinner(f"'{q}' 수집·학습 중..."):
                            try:
                                s, _log, _lk = crawl_topic(
                                    q, max_pages=auto_pages, return_sources=True,
                                    brave_api_key=auto_brave.strip() or None)
                            except Exception as e:
                                st.caption(f"'{q}' 수집 실패: {e}"); continue
                            if not s:
                                st.caption(f"'{q}' → 0문장"); 
                                stt.policy.record(q, -0.1)  # 수집 실패도 약한 벌점
                                continue
                            # 가드레일 통과분만
                            added, rej = stt.add_sentences(s, use_guardrail=True)
                            if added == 0:
                                st.caption(f"'{q}' → 전부 도메인 밖 거부")
                                stt.policy.record(q, -0.2)
                                continue
                            # 학습 전 지표
                            X = emb.encode_many(stt.corpus)
                            toks = [x.split() for x in stt.corpus]
                            m0 = measure(stt.gsom, X, toks)
                            # 학습
                            for _ in range(auto_rounds):
                                stt.gsom.round += 1
                                lr = max(0.02, 0.4 * (0.9 ** stt.gsom.round))
                                rad = max(0.5, 2.0 * (0.85 ** stt.gsom.round))
                                stt.gsom.train_step(X, lr, rad)
                                stt.gsom.grow()
                            m1 = measure(stt.gsom, X, toks)
                            m1["round"] = stt.gsom.round; m1["grew"] = 0
                            stt.history.append(m1)
                            # 보상 계산 + 정책 기록
                            reward, detail = learning_reward(
                                m0["mean_qe"], m1["mean_qe"],
                                m0["vocab_div"], m1["vocab_div"])
                            stt.policy.record(q, reward)
                            results.append((q, added, rej, reward, detail))
                    # 결과 표시
                    if results:
                        st.success(f"자율 학습 완료 · {len(results)}개 검색어")
                        for q, added, rej, reward, d in results:
                            tag = "✓" if reward > 0.25 else ("△" if reward > 0 else "✗")
                            st.markdown(
                                f"<div class='{'ok' if reward>0 else 'warn'}'>"
                                f"{tag} <b>{q}</b> · 보상 {reward:+.2f} "
                                f"(통과 {added}/거부 {rej}) · "
                                f"동화{d['assimilate']:+} 난이도{d['difficulty']:+} "
                                f"다양성{d['diversity']:+}</div>",
                                unsafe_allow_html=True)
                        st.caption("보상이 높았던 검색어 방향이 다음 자율 탐색에서 우선됩니다.")
        new_sentences = []  # 자율 모드는 위에서 학습까지 끝냄

    elif src == "코퍼스 직접 입력/업로드":
        c1, c2 = st.columns(2)
        txt = c1.text_area("문장 직접 입력 (줄바꿈으로 구분)", height=160,
                           placeholder="오늘 날씨가 좋아요\n밥을 먹어요\n...")
        files = c2.file_uploader(
            "또는 파일 업로드 (txt · md · docx · pdf, 여러 개 가능)",
            accept_multiple_files=True,
            key="corpusfiles",
            help="지원: txt, md, docx, pdf (다른 형식은 무시됩니다)",
        )
        if txt.strip():
            new_sentences += [s.strip() for s in txt.splitlines() if s.strip()]
        if files:
            from selfloop_engine import extract_corpus_from_file
            for f in files:
                tmp = f"/tmp/upload_{f.name}"
                with open(tmp, "wb") as out:
                    out.write(f.getbuffer())
                sents, info = extract_corpus_from_file(tmp, filename=f.name)
                if sents:
                    new_sentences += sents
                    c2.markdown(f'<div class="ok">{f.name}: {info}</div>',
                                unsafe_allow_html=True)
                else:
                    c2.markdown(f'<div class="warn">{f.name}: {info}</div>',
                                unsafe_allow_html=True)
            if new_sentences:
                c2.caption(f"추출 미리보기 ({len(new_sentences)}문장)")
                c2.code("\n".join(new_sentences[:6]), language=None)
    else:
        st.markdown("#### 🔍 검색 크롤링 학습")
        topic = st.text_input(
            "학습할 분야/검색어",
            placeholder="예: 발달장애 학생 재난 안전 쉬운말 / 온양초 교육계획 창체"
        )
        cbk1, cbk2 = st.columns([2, 1])
        brave_key = cbk1.text_input(
            "Brave Search API Key (권장)",
            value=os.environ.get("BRAVE_API_KEY", ""),
            type="password",
            help="https://api-dashboard.search.brave.com 에서 무료 발급(월 ~1000건). "
                 "키가 있으면 안정적으로 검색됩니다. 비우면 무료엔진 스크래핑(자주 차단)."
        )
        brave_bodies = cbk2.checkbox("본문까지 크롤링", value=True,
                                     help="켜면 검색된 URL 본문을 직접 수집(알참). "
                                          "끄면 검색 스니펫만 사용(빠름, API만으로 완결).")
        ca, cb, cc = st.columns(3)
        npg = ca.slider("검색 결과 페이지 수", 1, 20, 5)
        max_sents = cb.slider("최대 수집 문장", 30, 1000, 300, 10)
        delay = cc.slider("사이트 요청 간격(초)", 0.0, 2.0, 0.3, 0.1)

        with st.expander("고급: 직접 URL 추가 / 수집 정책", expanded=False):
            direct_urls = st.text_area(
                "직접 크롤링할 URL(줄바꿈)",
                placeholder="https://example.com/page1\nhttps://example.com/page2",
                height=90,
            )
            st.caption("공개 웹페이지의 텍스트만 수집합니다. 로그인이 필요한 페이지, 유료/저작권 문서, robots 정책이 민감한 사이트는 피하세요.")

        col_search, col_clear = st.columns([1, 1])
        if col_search.button("🔍 검색·크롤링해서 수집", type="primary"):
            urls = [u.strip() for u in direct_urls.splitlines() if u.strip()]
            with st.spinner(f"'{topic or '직접 URL'}' 검색·크롤링 중..."):
                try:
                    crawled, sources, links = crawl_topic(
                        topic,
                        max_pages=int(npg),
                        max_sentences=int(max_sents),
                        extra_urls=urls,
                        delay=float(delay),
                        return_sources=True,
                        brave_api_key=brave_key.strip() or None,
                        brave_fetch_bodies=bool(brave_bodies),
                    )
                    st.session_state._crawled = crawled
                    st.session_state._crawl_sources = sources
                    st.session_state._crawl_links = links
                    if crawled:
                        st.success(f"{len(crawled)}문장 수집 · 대상 URL {len(links)}개")
                    else:
                        diag = "<br>".join(str(s) for s in sources[:8])
                        st.markdown(
                            f'<div class="warn">수집된 문장 0건<br>{diag}</div>',
                            unsafe_allow_html=True,
                        )
                except Exception as e:
                    st.markdown(
                        f'<div class="warn">크롤링 실패: {e}<br>'
                        f'네트워크가 막힌 환경에서는 작동하지 않습니다. 로컬 PC에서 다시 실행하거나 URL을 직접 넣어보세요.</div>',
                        unsafe_allow_html=True,
                    )

        if col_clear.button("수집 결과 비우기"):
            st.session_state._crawled = []
            st.session_state._crawl_sources = []
            st.session_state._crawl_links = []
            st.rerun()

        new_sentences = st.session_state.get("_crawled", [])
        sources = st.session_state.get("_crawl_sources", [])
        links = st.session_state.get("_crawl_links", [])
        if links:
            st.caption("검색/크롤링 대상 URL")
            st.code("\n".join(links[:20]), language=None)
        if sources:
            with st.expander("수집 소스 로그", expanded=False):
                st.json(sources[:30])
        if new_sentences:
            st.caption(f"수집된 문장 미리보기 ({len(new_sentences)}개)")
            st.code("\n".join(new_sentences[:12]), language=None)
            st.download_button(
                "⬇ 수집 코퍼스 txt 다운로드",
                "\n".join(new_sentences).encode("utf-8"),
                file_name="crawled_corpus.txt",
                mime="text/plain",
            )

    st.markdown("### 2 · 라운드 옵션")
    c1, c2, c3 = st.columns(3)
    mode_r = c1.selectbox("종료 기준", ["라운드 수", "1분", "5분", "10분"])
    n_rounds = c2.number_input("라운드 수", 1, 200, 10, disabled=(mode_r != "라운드 수"))
    grow_on = c3.checkbox("GSOM 자율 성장", value=True)

    g1, g2 = st.columns(2)
    use_guard = g1.checkbox("🛡 마르코프 가드레일 (도메인 밖 문장 거부)", value=True,
                            help="새 문장을 코퍼스 도메인 기준으로 채점해, 임계값 밖이면 학습에서 제외. "
                                 "광고·잡음·이질 도메인을 자동으로 거른다.")
    guard_pct = g2.slider("가드레일 엄격도(높을수록 엄격)", 5, 40, 25,
                          disabled=not use_guard,
                          help="높일수록 더 많은 문장을 거부(엄격). 잡음 섞인 코퍼스는 25 이상 권장.")

    if st.button("▶ 학습 시작", type="primary"):
        # 가드레일 적용: 기존 코퍼스로 학습 후 새 문장 필터링
        guard_msg = ""
        if new_sentences:
            if use_guard and len(stt.corpus) >= 10:
                stt.fit_guardrail(percentile=guard_pct)
                added, rej = stt.add_sentences(new_sentences, use_guardrail=True)
                guard_msg = (f"가드레일: {added}문장 통과 · {rej}문장 거부 "
                             f"(임계값 {stt.guard_threshold:.2f})")
            else:
                stt.corpus += new_sentences
                guard_msg = f"{len(new_sentences)}문장 추가 (가드레일 미적용: 코퍼스 부족 또는 꺼짐)"
        if not stt.corpus:
            st.markdown('<div class="warn">학습할 코퍼스가 없습니다.</div>',
                        unsafe_allow_html=True)
        else:
            if guard_msg:
                st.markdown(f'<div class="ok">{guard_msg}</div>', unsafe_allow_html=True)
            X = emb.encode_many(stt.corpus)
            # --- 차원 불일치 방어 ---
            emb_dim = X.shape[1]
            som_dim = stt.gsom.W.shape[1] if stt.gsom.W.size else emb_dim
            if emb_dim != som_dim:
                st.markdown(
                    f'<div class="warn">차원 불일치: 임베딩 dim({emb_dim}) ≠ '
                    f'SOM dim({som_dim}). 현재 임베딩에 맞춰 SOM을 새로 시작합니다. '
                    f'(이전 SOM 학습은 초기화되지만 코퍼스는 유지됩니다)</div>',
                    unsafe_allow_html=True)
                from selfloop_engine import GrowingSOM
                import numpy as _np
                stt.dim = emb_dim
                stt.gsom = GrowingSOM(dim=emb_dim, init_nodes=36, seed=0)
                _rng = _np.random.default_rng(0)
                _idx = _rng.choice(len(X), min(36, len(X)), replace=False)
                stt.gsom.W = X[_idx].copy() + _rng.normal(scale=0.01, size=(len(_idx), emb_dim))
                stt.gsom.coords = stt.gsom.coords[:len(_idx)]
                stt.gsom.err = stt.gsom.err[:len(_idx)]
                stt.history = []
            toks = [s.split() for s in stt.corpus]
            time_limit = {"1분":60,"5분":300,"10분":600}.get(mode_r)
            t_start = time.time()
            prog = st.progress(0.0)
            live = st.empty()
            r = 0
            while True:
                r += 1
                stt.gsom.round += 1
                lr = 0.4 * (0.9 ** stt.gsom.round)
                rad = max(0.5, 2.0 * (0.85 ** stt.gsom.round))
                stt.gsom.train_step(X, lr, rad)
                grew = stt.gsom.grow() if grow_on else 0
                m = measure(stt.gsom, X, toks)
                m["round"] = stt.gsom.round; m["grew"] = grew
                stt.history.append(m)
                w = collapse_warning(stt.history)
                live.markdown(
                    f"<div class='{'warn' if w else 'ok'}'>r{stt.gsom.round} · "
                    f"nodes={m['nodes']} occ={m['occupancy']} "
                    f"qe={m['mean_qe']:.2f} vocab={m['vocab_div']:.2f}"
                    f"{' · '+w if w else ''}</div>", unsafe_allow_html=True)
                if time_limit:
                    prog.progress(min(1.0, (time.time()-t_start)/time_limit))
                    if time.time()-t_start >= time_limit: break
                else:
                    prog.progress(min(1.0, r/n_rounds))
                    if r >= n_rounds: break
            st.success(f"학습 완료 · 총 {r}라운드 · 노드 {stt.gsom.n}")
            if stt.rejected_log:
                with st.expander(f"거부된 문장 보기 ({len(stt.rejected_log)})"):
                    for s, sc in stt.rejected_log[-30:]:
                        st.caption(f"[{sc}] {s[:80]}")

    # ---- 계측 차트 ----
    if stt.history:
        st.markdown("### 3 · 계측")
        h = stt.history
        rounds = [x["round"] for x in h]
        c1,c2,c3,c4 = st.columns(4)
        last = h[-1]
        metric(c1,"NODES", last["nodes"])
        metric(c2,"OCCUPANCY", last["occupancy"])
        metric(c3,"MEAN QE", f"{last['mean_qe']:.2f}")
        metric(c4,"VOCAB DIV", f"{last['vocab_div']:.2f}")

        import pandas as pd
        df = pd.DataFrame(h).set_index("round")
        cc1, cc2 = st.columns(2)
        cc1.caption("점유율 / 노드")
        cc1.line_chart(df[["occupancy","nodes"]])
        cc2.caption("평균 QE / 어휘다양성")
        cc2.line_chart(df[["mean_qe","vocab_div"]])

        w = collapse_warning(h)
        if w:
            st.markdown(f'<div class="warn">⚠ {w} — 인간 피드백 필요 시점</div>',
                        unsafe_allow_html=True)

# ======================================================================
# PAGE 2 — 답변 루프
# ======================================================================
with page[1]:
    stt = st.session_state.state
    emb = st.session_state.emb

    if not stt.corpus:
        st.markdown('<div class="warn">먼저 학습 루프에서 코퍼스를 학습하거나 '
                    '사이드바에서 pkl을 불러오세요.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f"### 학습된 지도에 질문하기")
        st.caption(f"코퍼스 {len(stt.corpus)}문장 · {stt.gsom.n}노드 위에서 답변")

        c1, c2, c3 = st.columns(3)
        model_options = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o", "gpt-4.1", "gpt-4-turbo", "직접 입력"]
        model_pick = c1.selectbox("LLM 모델", model_options)
        model = c1.text_input("모델 직접 입력", value="gpt-4o-mini") if model_pick == "직접 입력" else model_pick
        temp = c2.slider("temperature", 0.0, 1.0, 0.3, 0.1)
        topk = c3.slider("맥락 문장 수", 3, 20, 10)
        save_qa = st.checkbox("질문·답변을 코퍼스에 저장(자기성찰 누적)", value=True)

        q = st.text_input("질문", placeholder="학습한 내용을 바탕으로 물어보세요")

        if st.button("▶ 답변 생성", type="primary") and q.strip():
            qv = emb.encode(q)
            # 차원 불일치 방어: 임베딩과 SOM dim이 다르면 안내 후 중단
            if stt.gsom.W.size and qv.shape[0] != stt.gsom.W.shape[1]:
                st.markdown(
                    f'<div class="warn">차원 불일치: 임베딩 dim({qv.shape[0]}) ≠ '
                    f'SOM dim({stt.gsom.W.shape[1]}).<br>학습 루프에서 현재 임베딩으로 '
                    f'다시 학습한 뒤 질문하세요(또는 저장 당시 임베딩을 올리세요).</div>',
                    unsafe_allow_html=True)
                st.stop()
            # SOM 가드레일: 질문이 학습 분포 안인가 (BMU 거리)
            qbmu, qdist = stt.gsom.bmu(qv)
            # 코퍼스에서 질문 BMU에 가까운 문장들 = 맥락
            Xc = emb.encode_many(stt.corpus)
            sims = [-np.linalg.norm(emb.encode(s) - qv) for s in stt.corpus]
            order = np.argsort(sims)[::-1][:topk]
            context = [stt.corpus[i] for i in order]

            # 가드레일 판정
            qe_vals = [h["mean_qe"] for h in stt.history] or [qdist]
            band = float(np.mean(qe_vals) + 2*np.std(qe_vals)) if len(qe_vals)>1 else qdist*1.5
            in_domain = qdist <= band

            cL, cR = st.columns([3,2])
            with cL:
                if in_domain:
                    st.markdown(f'<div class="ok">✓ 도메인 내 질문 '
                                f'(BMU거리 {qdist:.2f} ≤ 경계 {band:.2f})</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="warn">⚠ 도메인 이탈 의심 '
                                f'(BMU거리 {qdist:.2f} > 경계 {band:.2f}) — '
                                f'학습 범위 밖일 수 있음</div>', unsafe_allow_html=True)

                with st.spinner("LLM 호출 중..."):
                    ans = llm_answer(
                        q,
                        context,
                        model=model,
                        temperature=temp,
                        max_tokens=int(max_tokens_input),
                        api_key=api_key_input.strip() or None,
                        base_url=base_url_input.strip() or None,
                    )
                st.markdown("#### 답변")
                st.write(ans)

                if save_qa and ans and not ans.startswith("["):
                    stt.corpus.append(f"질문: {q}")
                    stt.corpus.append(f"답변: {ans}")
                    st.info("질문·답변을 코퍼스에 저장했습니다. 다음 학습 라운드에서 의미지도에 반영됩니다.")

            with cR:
                st.markdown("#### 사용된 맥락")
                st.markdown('<div class="pathbox">'+"<br>".join(
                    f"· {c}" for c in context[:8])+'</div>',
                    unsafe_allow_html=True)
                # 추론경로: 질문 BMU -> 답변에 가장 가까운 코퍼스 BMU
                abmu, _ = stt.gsom.bmu(emb.encode(context[0]))
                path, cost, hops = stt.gsom.semantic_path(qbmu, abmu)
                st.markdown("#### 추론경로 (논리의 발자국)")
                st.markdown(f'<div class="pathbox">질문격자 #{qbmu} → 답변격자 #{abmu}<br>'
                            f'의미 도정거리 = <b style="color:#2dd4bf">{cost:.2f}</b><br>'
                            f'거친 노드 = <b style="color:#2dd4bf">{hops}</b><br>'
                            f'경로: {" → ".join("#"+str(p) for p in path[:10])}'
                            f'{" ..." if len(path)>10 else ""}</div>',
                            unsafe_allow_html=True)

        st.divider()
        st.caption("팁: 답변 저장을 켜면 Q·A가 코퍼스에 누적됩니다. 이후 학습 루프를 다시 돌리면 SOM 지도와 도메인 경계에 반영됩니다.")


# ======================================================================
# PAGE 3 — 정밀 학습 (부분 역양자화 + 소프트맥스 + 국소 미세조정)
# ======================================================================
with page[2]:
    stt = st.session_state.state
    emb = st.session_state.emb
    st.markdown("### ⚡ 정밀 학습 (관심 영역 집중)")
    st.caption("관심 있는 주제(여러 문장)만 골라, SOM의 그 영역만 역양자화해서 "
               "소프트맥스로 부드럽게 미세조정합니다. 전체 재학습 없이 관심사만 또렷해집니다. "
               "※ 한 문장만 넣으면 과적합됩니다 — 관련 문장 여러 개(영역)를 넣으세요.")

    if not stt.corpus or stt.gsom.W.size == 0:
        st.markdown('<div class="warn">먼저 학습 루프에서 코퍼스를 학습하세요.</div>',
                    unsafe_allow_html=True)
    else:
        region_text = st.text_area(
            "관심 영역 문장들 (줄바꿈으로 구분, 2개 이상 권장)",
            height=120, placeholder="예)\n뜀틀 도약 기술\n매트 구르기 동작\n평균대 균형 잡기")
        c1, c2 = st.columns(2)
        passes = c1.slider("미세조정 반복", 1, 15, 8)
        do_forget = c2.checkbox("망각 적용 (안 쓴 영역 흐리기)", value=False)

        if st.button("⚡ 정밀 학습 실행", type="primary"):
            from soft_refine import SoftRefiner
            from selfloop_engine import Quantizer
            lines = [s.strip() for s in region_text.split("\n") if s.strip()]
            if len(lines) < 1:
                st.markdown('<div class="warn">관심 문장을 입력하세요.</div>',
                            unsafe_allow_html=True)
            else:
                if len(lines) == 1:
                    st.markdown('<div class="warn">한 문장만 넣으면 과적합될 수 있습니다. '
                                '관련 문장을 여러 개 넣는 것을 권장합니다.</div>',
                                unsafe_allow_html=True)
                qz = Quantizer(stt.gsom.W)
                Wq = qz.q(stt.gsom.W)
                ref = SoftRefiner(Wq, qz.scale, temp=2.0, lr=0.3)
                region_vecs = [emb.encode(s) for s in lines]
                qe_before = float(np.mean([ref.local_qe(v) for v in region_vecs]))
                params = ref.refine_region(region_vecs, passes=passes)
                qe_after = float(np.mean([ref.local_qe(v) for v in region_vecs]))
                forget_info = ref.forget() if do_forget else None
                # 미세조정 결과를 SOM에 반영 (역양자화해서 W 갱신)
                stt.gsom.W = ref.W_q.astype(np.float64) * qz.scale
                msg = (f"정밀 학습 완료 · 영역 QE {qe_before:.3f} → {qe_after:.3f} "
                       f"({(qe_before-qe_after)/max(qe_before,1e-6)*100:+.1f}%) · "
                       f"자동 파라미터: temp={params['temp']}, lr={params['lr']}, k={params['k']}")
                st.markdown(f'<div class="ok">{msg}</div>', unsafe_allow_html=True)
                if forget_info:
                    st.caption(f"망각: {forget_info['forgotten']}개 노드 흐림 처리, "
                               f"{forget_info['kept_sharp']}개 또렷 유지")


# ======================================================================
# PAGE 4 — 연속 자동학습 (스스로 검색하며 다양한 분야 학습)
# ======================================================================
with page[3]:
    stt = st.session_state.state
    emb = st.session_state.emb
    st.markdown("### ♾️ 연속 자동학습")
    st.caption("대화/씨앗 주제에서 출발해, 스스로 검색어를 만들고 Brave로 검색·수집·학습을 "
               "반복합니다. 보상이 높았던 방향은 강화하고 가끔 새 분야로 탐색합니다. "
               "한 사이클씩 돌며 결과를 보여줍니다.")

    if len(stt.corpus) < 5:
        st.markdown('<div class="warn">먼저 어느 정도(5문장 이상) 학습된 코퍼스가 필요합니다.</div>',
                    unsafe_allow_html=True)
    else:
        a1, a2, a3 = st.columns(3)
        al_brave = a1.text_input("Brave API Key", type="password",
                                 value=os.environ.get("BRAVE_API_KEY", ""),
                                 key="al_brave")
        n_cycles = a2.slider("실행할 사이클 수", 1, 10, 3)
        al_pages = a3.slider("사이클당 페이지", 1, 5, 3)
        seed = st.text_input("씨앗 주제 (비우면 SOM이 스스로 방향 선택)",
                             placeholder="예) 특수교육 개별화 / 비우면 자동")

        if st.button("♾️ 연속 자동학습 시작", type="primary"):
            from auto_learn import AutoLearner
            learner = AutoLearner(stt, emb,
                                  brave_api_key=al_brave.strip() or None,
                                  max_pages=al_pages, train_rounds=3)
            prog = st.progress(0.0)
            box = st.container()
            for i in range(n_cycles):
                rec = learner.step(seed_text=seed.strip() or None)
                prog.progress((i + 1) / n_cycles)
                tag = "✅" if (rec.get("reward") or 0) > 0 else "⚠️"
                with box:
                    rw = rec.get("reward")
                    st.markdown(
                        f"<div class='{'ok' if (rw or 0)>0 else 'warn'}'>"
                        f"{tag} <b>사이클 {rec['cycle']}</b> · 검색어: <b>{rec['query']}</b> "
                        f"({rec['mode']}) · 추가 {rec['added']}/거부 {rec['rejected']} · "
                        f"보상 {rw if rw is not None else '-'} · {rec['note']}</div>",
                        unsafe_allow_html=True)
            st.success(f"{n_cycles}사이클 완료 · 현재 코퍼스 {len(stt.corpus)}문장 · "
                       f"노드 {stt.gsom.n}")
            st.caption("보상이 높았던 검색어 방향이 다음 실행에서 우선됩니다. "
                       "여러 번 돌릴수록 잘 배워지는 분야로 수렴하고, 가끔 새 분야로 넓힙니다.")
