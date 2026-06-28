"""
app.py — GSOM 제품용 UI (Streamlit)

설계:
- 흰 배경, 깔끔한 카드형
- 사이드바: OpenAI 키, pkl 저장/불러오기, (고급) Brave 키
- 학습 탭: 자료 업로드 + 그래프
- 대화 탭: 카카오톡 스타일 (내 질문=연회색 오른쪽, 답변=파란색 왼쪽)
- 기본 모드: 키1개+학습+대화 / 고급 모드: Brave 검색·자동학습·정밀학습
- tok_emb 자동 생성 (사용자는 신경 안 씀)
"""
import os
import streamlit as st
import numpy as np

from selfloop_engine import (SelfLoopState, EmbeddingProvider, GrowingSOM,
                             measure, collapse_warning, extract_corpus_from_file,
                             build_tok_emb_from_corpus, llm_answer)

st.set_page_config(page_title="GSOM AI", page_icon="●", layout="wide")

# ---------------- 스타일 (흰 배경, 카톡 말풍선) ----------------
st.markdown("""
<style>
.stApp { background: #ffffff; }
.block-container { padding-top: 1.5rem; max-width: 900px; }
/* 카톡 말풍선 */
.chat-row { display: flex; margin: 6px 0; }
.chat-row.me { justify-content: flex-end; }
.chat-row.ai { justify-content: flex-start; }
.bubble {
  max-width: 78%; padding: 10px 14px; border-radius: 16px;
  font-size: 15px; line-height: 1.5; word-break: break-word; white-space: pre-wrap;
}
.bubble.me { background: #ededed; color: #111; border-bottom-right-radius: 4px; }
.bubble.ai { background: #3897f0; color: #fff; border-bottom-left-radius: 4px; }
.meta { font-size: 11px; color: #999; margin: 2px 6px; }
.ok { background:#e8f5e9; color:#2e7d32; padding:8px 12px; border-radius:8px; }
.warn { background:#fff3e0; color:#e65100; padding:8px 12px; border-radius:8px; }
</style>
""", unsafe_allow_html=True)

DIM = 64

# ---------------- 세션 상태 ----------------
if "state" not in st.session_state:
    st.session_state.state = SelfLoopState(dim=DIM)
if "emb" not in st.session_state:
    st.session_state.emb = EmbeddingProvider(dim=DIM)
if "chat" not in st.session_state:
    st.session_state.chat = []   # [(role, text, meta)]
if "advanced" not in st.session_state:
    st.session_state.advanced = False

stt = st.session_state.state
emb = st.session_state.emb

# ======================== 사이드바 ========================
with st.sidebar:
    st.markdown("### ⚙️ 설정")
    openai_key = st.text_input("OpenAI API Key", type="password",
                               value=os.environ.get("OPENAI_API_KEY", ""))
    st.session_state.advanced = st.toggle("고급 모드", value=st.session_state.advanced)
    brave_key = ""
    if st.session_state.advanced:
        brave_key = st.text_input("Brave API Key (검색용)", type="password",
                                  value=os.environ.get("BRAVE_API_KEY", ""))

    st.divider()
    st.markdown("##### 모델 저장 / 불러오기")
    up = st.file_uploader("불러오기 (.pkl)", type=None, key="pkl_up")
    if up is not None:
        try:
            import pickle, tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
                f.write(up.getvalue()); tmp = f.name
            st.session_state.state = SelfLoopState.load(tmp)
            stt = st.session_state.state
            # 불러온 코퍼스로 tok_emb 자동 재생성 (의미 임베딩 복원)
            if len(stt.corpus) >= 50:
                try:
                    mat, w2i = build_tok_emb_from_corpus(stt.corpus, dim=DIM, epochs=8)
                    if mat is not None:
                        st.session_state.emb.apply_matrix(mat, w2i)
                        emb = st.session_state.emb
                except Exception:
                    pass
            # TF-IDF 검색 인덱스 재구축 (불러오자마자 대화 가능)
            from selfloop_engine import CorpusSearcher
            st.session_state.searcher = CorpusSearcher().build(stt.corpus)
            st.markdown('<div class="ok">불러왔습니다.</div>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f'<div class="warn">불러오기 실패: {e}</div>', unsafe_allow_html=True)

    if st.button("현재 상태 저장 (.pkl 생성)"):
        try:
            path = "/tmp/gsom_state.pkl"
            stt.save(path)
            with open(path, "rb") as f:
                st.download_button("⬇️ 다운로드", f.read(),
                                   file_name="gsom_state.pkl")
        except Exception as e:
            st.markdown(f'<div class="warn">저장 실패: {e}</div>', unsafe_allow_html=True)

    st.divider()
    st.caption(f"코퍼스 {len(stt.corpus)}문장 · 노드 {stt.gsom.n} · 임베딩 {emb.mode}")

# ======================== 메인 ========================
st.markdown("## ● GSOM AI")

tabs = st.tabs(["💬 대화", "📚 학습"] +
               (["⚡ 정밀학습", "♾️ 자동학습"] if st.session_state.advanced else []))

# ---------------- 대화 탭 (카톡 스타일) ----------------
with tabs[0]:
    # 전체 대화를 하나의 스크롤 박스로 렌더 (최신이 아래에 보이도록 자동 스크롤)
    import html as _html
    rows = ""
    for role, text, meta in st.session_state.chat:
        side = "me" if role == "me" else "ai"
        safe = _html.escape(text).replace("\n", "<br>")
        rows += f'<div class="chat-row {side}"><div class="bubble {side}">{safe}</div></div>'
        if meta:
            align = "right" if side == "me" else "left"
            rows += f'<div class="meta" style="text-align:{align}">{_html.escape(meta)}</div>'
    chat_html = f"""
<div id="chatbox" style="height:60vh;overflow-y:auto;padding:8px 4px;
     border:1px solid #eee;border-radius:12px;background:#fafafa;">
  {rows if rows else '<div style="color:#bbb;text-align:center;padding:2rem;">대화를 시작하세요</div>'}
</div>
<script>
  var cb = document.getElementById('chatbox');
  if (cb) {{ cb.scrollTop = cb.scrollHeight; }}
</script>
"""
    import streamlit.components.v1 as components
    components.html(
        "<style>"
        ".chat-row{display:flex;margin:6px 0;}"
        ".chat-row.me{justify-content:flex-end;}"
        ".chat-row.ai{justify-content:flex-start;}"
        ".bubble{max-width:78%;padding:10px 14px;border-radius:16px;"
        "font-size:15px;line-height:1.5;word-break:break-word;}"
        ".bubble.me{background:#ededed;color:#111;border-bottom-right-radius:4px;}"
        ".bubble.ai{background:#3897f0;color:#fff;border-bottom-left-radius:4px;}"
        ".meta{font-size:11px;color:#999;margin:2px 6px;}"
        "body{margin:0;font-family:'Noto Sans KR',sans-serif;}"
        "</style>" + chat_html,
        height=480, scrolling=False)

    q = st.chat_input("메시지를 입력하세요")
    if q:
        st.session_state.chat.append(("me", q, ""))
        # 도메인 판단
        domain_ok, score = True, 0.0
        if stt.guardrail is not None and stt.guardrail.trained:
            domain_ok, score = stt.guardrail.judge(q, stt.guard_threshold)
        if not stt.corpus:
            ans = "아직 학습된 자료가 없습니다. '학습' 탭에서 자료를 넣어주세요."
            meta = ""
        elif not domain_ok:
            ans = "학습된 분야 밖의 질문으로 보여 답변을 보류합니다."
            meta = f"도메인 점수 {score:.1f}"
        else:
            # 하이브리드 검색(TF-IDF + tok_emb)으로 관련 맥락 추출
            if st.session_state.get("searcher") is None:
                from selfloop_engine import CorpusSearcher
                st.session_state.searcher = CorpusSearcher().build(stt.corpus)
            hits = st.session_state.searcher.search_hybrid(q, emb=emb, topk=8)
            ctx = [s for s, _ in hits]
            if openai_key.strip():
                try:
                    ans = llm_answer(q, ctx, api_key=openai_key.strip(),
                                     max_tokens=1500)
                except Exception as e:
                    ans = f"(LLM 호출 오류: {e})"
                meta = f"하이브리드 검색 {len(ctx)}문장 + LLM"
            else:
                ans = "관련 자료:\n" + "\n".join(f"· {c}" for c in ctx[:5])
                meta = "검색 단독 (OpenAI 키 없음)"
        st.session_state.chat.append(("ai", ans, meta))
        st.rerun()

# ---------------- 학습 탭 ----------------
with tabs[1]:
    st.markdown("##### 자료 넣기")
    files = st.file_uploader("txt / pdf / docx (여러 개 가능)",
                             accept_multiple_files=True, key="train_up")
    if files and st.button("자료 추가"):
        total = 0
        infos = []
        for f in files:
            try:
                import tempfile
                suffix = os.path.splitext(f.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as t:
                    t.write(f.getvalue()); tmp = t.name
                sents, info = extract_corpus_from_file(tmp, f.name)
                if not sents:
                    st.markdown(f'<div class="warn">{f.name}: {info}</div>',
                                unsafe_allow_html=True)
                    continue
                added, _ = stt.add_sentences(sents, use_guardrail=False)
                total += added
                infos.append(f"{f.name}: {info} → 추가 {added}")
            except Exception as e:
                st.markdown(f'<div class="warn">{f.name}: {e}</div>',
                            unsafe_allow_html=True)
        st.markdown(f'<div class="ok">{total}문장 추가 (중복 제외)</div>',
                    unsafe_allow_html=True)
        for line in infos:
            st.caption(line)

    if st.button("▶ 학습 실행", type="primary"):
        if not stt.corpus:
            st.markdown('<div class="warn">먼저 자료를 추가하세요.</div>',
                        unsafe_allow_html=True)
        else:
            with st.spinner("tok_emb 자동 생성 + 학습 중..."):
                # 1) tok_emb 자동 생성 (자료 충분하면)
                if len(stt.corpus) >= 50:
                    mat, w2i = build_tok_emb_from_corpus(stt.corpus, dim=DIM, epochs=8)
                    if mat is not None:
                        emb.apply_matrix(mat, w2i)
                # 2) SOM 학습
                X = emb.encode_many(stt.corpus)
                ed = X.shape[1]
                sd = stt.gsom.W.shape[1] if stt.gsom.W.size else ed
                if ed != sd:
                    stt.gsom = GrowingSOM(dim=ed, init_nodes=min(36, len(X)), seed=0)
                    rng = np.random.default_rng(0)
                    idx = rng.choice(len(X), min(36, len(X)), replace=False)
                    stt.gsom.W = X[idx].copy() + rng.normal(scale=0.01, size=(len(idx), ed))
                    stt.gsom.coords = stt.gsom.coords[:len(idx)]
                    stt.gsom.err = stt.gsom.err[:len(idx)]
                toks = [s.split() for s in stt.corpus]
                for r in range(40):
                    stt.gsom.round += 1
                    lr = max(0.02, 0.4 * (0.9 ** stt.gsom.round))
                    rad = max(0.5, 2.0 * (0.85 ** stt.gsom.round))
                    stt.gsom.train_step(X, lr, rad)
                    stt.gsom.grow()
                    stt.history.append({**measure(stt.gsom, X, toks),
                                        "round": stt.gsom.round})
                if len(stt.corpus) >= 10:
                    stt.fit_guardrail(percentile=25)
                # TF-IDF 검색 인덱스 재구축
                from selfloop_engine import CorpusSearcher
                st.session_state.searcher = CorpusSearcher().build(stt.corpus)
            warn = collapse_warning(stt.history)
            m = stt.history[-1]
            msg = f"학습 완료 · QE {m['mean_qe']:.2f} · 노드 {stt.gsom.n} · 임베딩 {emb.mode}"
            st.markdown(f'<div class="ok">{msg}</div>', unsafe_allow_html=True)
            if warn:
                st.markdown(f'<div class="warn">{warn}</div>', unsafe_allow_html=True)

    # 학습 그래프
    if stt.history:
        import pandas as pd
        df = pd.DataFrame(stt.history)
        st.markdown("##### 학습 추이")
        c1, c2 = st.columns(2)
        with c1:
            st.line_chart(df.set_index("round")[["mean_qe"]], height=200)
            st.caption("QE (낮을수록 수렴)")
        with c2:
            st.line_chart(df.set_index("round")[["vocab_div", "occ_ratio"]], height=200)
            st.caption("어휘다양성 / 점유율")

# ---------------- 고급: 정밀학습 ----------------
if st.session_state.advanced and len(tabs) > 2:
    with tabs[2]:
        st.markdown("##### ⚡ 관심 영역 정밀학습")
        st.caption("관련 문장 여러 개(영역)를 넣으면 그 영역만 또렷하게 미세조정합니다.")
        region_text = st.text_area("관심 문장들 (줄바꿈 구분, 2개 이상)", height=110)
        if st.button("정밀학습 실행"):
            from soft_refine import SoftRefiner
            from selfloop_engine import Quantizer
            lines = [s.strip() for s in region_text.split("\n") if s.strip()]
            if not stt.gsom.W.size or not lines:
                st.markdown('<div class="warn">학습된 모델과 관심 문장이 필요합니다.</div>',
                            unsafe_allow_html=True)
            else:
                qz = Quantizer(stt.gsom.W)
                ref = SoftRefiner(qz.q(stt.gsom.W), qz.scale, temp=2.0, lr=0.3)
                vecs = [emb.encode(s) for s in lines]
                qe0 = float(np.mean([ref.local_qe(v) for v in vecs]))
                p = ref.refine_region(vecs, passes=8)
                qe1 = float(np.mean([ref.local_qe(v) for v in vecs]))
                stt.gsom.W = ref.W_q.astype(np.float64) * qz.scale
                st.markdown(f'<div class="ok">영역 QE {qe0:.2f} → {qe1:.2f} '
                            f'(temp={p["temp"]}, k={p["k"]})</div>',
                            unsafe_allow_html=True)

# ---------------- 고급: 자동학습 ----------------
if st.session_state.advanced and len(tabs) > 3:
    with tabs[3]:
        st.markdown("##### ♾️ 연속 자동학습 (Brave 검색)")
        if not brave_key.strip():
            st.markdown('<div class="warn">사이드바에 Brave API Key를 입력하세요.</div>',
                        unsafe_allow_html=True)
        else:
            seed = st.text_input("씨앗 주제 (비우면 자동)")
            n_cyc = st.slider("사이클 수", 1, 8, 3)
            if st.button("자동학습 시작"):
                from auto_learn import AutoLearner
                learner = AutoLearner(stt, emb, brave_api_key=brave_key.strip(),
                                      max_pages=3, train_rounds=3)
                prog = st.progress(0.0)
                for i in range(n_cyc):
                    rec = learner.step(seed_text=seed.strip() or None)
                    prog.progress((i + 1) / n_cyc)
                    rw = rec.get("reward")
                    cls = "ok" if (rw or 0) > 0 else "warn"
                    st.markdown(f'<div class="{cls}">사이클 {rec["cycle"]} · '
                                f'{rec["query"]} · 추가 {rec["added"]} · 보상 {rw} · '
                                f'{rec["note"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="ok">완료 · 코퍼스 {len(stt.corpus)}문장</div>',
                            unsafe_allow_html=True)
