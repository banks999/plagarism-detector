"""
app.py - Streamlit demo for the text similarity / plagiarism detector.

    streamlit run app.py
"""
from __future__ import annotations

import html
import os

import pandas as pd
import streamlit as st

import data
from engine import METHODS, Detector, EmbedderUnavailable, load_embedder, split_sentences

BACKEND = os.environ.get("PLAGIARISM_BACKEND", "auto")  # "lsa" = offline test stand-in (not for real use)
METHOD_LABELS = {
    "tfidf": "TF-IDF cosine",
    "containment": "3-gram containment",
    "semantic": "Semantic (SBERT)",
    "combined": "Combined",
}
KIND_STYLE = {
    "verbatim": ("#f4a6a6", "Verbatim copy"),
    "edited": ("#f9cf9a", "Edited copy"),
    "paraphrase": ("#f6e58d", "Paraphrase (same meaning, different words)"),
}

st.set_page_config(page_title="Plagiarism & Text Similarity Detector", page_icon="🔍", layout="wide")


@st.cache_resource(show_spinner="Loading the sentence-transformer model (the first run downloads ~90 MB)...")
def get_embedder(backend: str):
    return load_embedder(backend, lsa_corpus=data.all_sentences())


def highlight_suspect(text, sentences) -> str:
    parts, pos = [], 0
    for s in sentences:
        parts.append(html.escape(text[pos : s.start]))
        seg = html.escape(text[s.start : s.end])
        if s.flagged:
            colour, label = KIND_STYLE[s.kind]
            tip = html.escape(f"{label}. Matches: {s.source_text}  [combined {s.scores['combined']:.2f}]", quote=True)
            parts.append(
                f'<mark style="background:{colour};color:#111;padding:1px 3px;border-radius:3px" title="{tip}">{seg}</mark>'
            )
        else:
            parts.append(seg)
        pos = s.end
    parts.append(html.escape(text[pos:]))
    return '<div style="white-space:pre-wrap;line-height:1.75">' + "".join(parts) + "</div>"


def highlight_source(sources: dict, matched: set) -> str:
    blocks = []
    for name, text in sources.items():
        parts, pos = [], 0
        for sp in split_sentences(text):
            parts.append(html.escape(text[pos : sp.start]))
            seg = html.escape(sp.text)
            if sp.text in matched:
                parts.append(f'<mark style="background:#a9d3f5;color:#111;padding:1px 3px;border-radius:3px">{seg}</mark>')
            else:
                parts.append(seg)
            pos = sp.end
        parts.append(html.escape(text[pos:]))
        header = f"<div style='font-weight:600;margin-bottom:4px'>{html.escape(name)}</div>" if len(sources) > 1 else ""
        blocks.append(f'{header}<div style="white-space:pre-wrap;line-height:1.75;margin-bottom:14px">{"".join(parts)}</div>')
    return "".join(blocks)


def load_demo(i: int):
    demo = data.demo_samples()[i]
    st.session_state["source_text"] = demo["source"]
    st.session_state["suspect_text"] = demo["suspect"]
    st.session_state["ran"] = True


# ------------------------------------------------------------------------------------ model
try:
    embedder = get_embedder(BACKEND)
except EmbedderUnavailable as e:
    st.title("🔍 Plagiarism & Text Similarity Detector")
    st.error(str(e))
    st.stop()

base = Detector(embedder)

# ------------------------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    combined_thr = st.slider(
        "Combined-score threshold",
        0.20, 0.95, float(base.thresholds["combined"]), 0.01,
        help="A suspect sentence is highlighted when its combined score reaches this value. Lower = more sensitive.",
    )
    st.caption(f"Model: `{base.backend}`")
    thr_rows = [
        {"Signal": METHOD_LABELS[m], "Threshold": base.thresholds[m] if m != "combined" else combined_thr} for m in METHODS
    ]
    st.dataframe(pd.DataFrame(thr_rows), hide_index=True, width="stretch")
    st.caption(f"A document is called suspicious when at least {base.thresholds['coverage']:.0%} of its text is flagged.")
    st.divider()
    st.markdown(
        "**How it works**\n\n"
        "Each suspect sentence is compared with every source sentence using:\n"
        "- **TF-IDF cosine** (shared wording, weighted by rarity)\n"
        "- **3-gram containment** (verbatim runs of words)\n"
        "- **SBERT embeddings** (shared *meaning*)\n"
        "- a **combined** score of the three"
    )

det = Detector(embedder, thresholds={**base.thresholds, "combined": combined_thr})

# ------------------------------------------------------------------------------------ header
st.title("🔍 Plagiarism & Text Similarity Detector")
st.caption("Compare a suspect text against one or more source texts. Lexical methods find copying; the semantic layer also finds paraphrasing.")

if BACKEND == "lsa":
    st.warning("TEST MODE: the offline stand-in is active, **not SBERT**. Do not present these results as neural.")
elif not base.thresholds_tuned:
    st.info("Using placeholder thresholds. Run `python evaluate.py` once to tune them for this model; this page then picks them up automatically.")

st.write("**Try a demo case** (all use the same source passage):")
demo_cols = st.columns(len(data.demo_samples()))
for i, (col, demo) in enumerate(zip(demo_cols, data.demo_samples())):
    col.button(demo["name"], key=f"demo{i}", on_click=load_demo, args=(i,), width="stretch")

# ------------------------------------------------------------------------------------ inputs
left, right = st.columns(2)
with left:
    st.subheader("Source text")
    st.text_area("Paste the original / reference text", key="source_text", height=230, label_visibility="collapsed",
                 placeholder="Paste the original text here...")
    source_files = st.file_uploader("...or upload one or more .txt / .md files", type=["txt", "md"], accept_multiple_files=True)
with right:
    st.subheader("Suspect text")
    st.text_area("Paste the text to check", key="suspect_text", height=230, label_visibility="collapsed",
                 placeholder="Paste the text you want to check here...")
    suspect_file = st.file_uploader("...or upload a .txt / .md file (replaces the pasted text)", type=["txt", "md"])

if st.button("Check for plagiarism", type="primary"):
    st.session_state["ran"] = True

sources = {}
if st.session_state.get("source_text", "").strip():
    sources["pasted source"] = st.session_state["source_text"]
for f in source_files or []:
    sources[f.name] = f.read().decode("utf-8", errors="ignore")
suspect = suspect_file.read().decode("utf-8", errors="ignore") if suspect_file else st.session_state.get("suspect_text", "")

if not st.session_state.get("ran"):
    st.stop()
if not sources or not suspect.strip():
    st.warning("Provide both a source text and a suspect text.")
    st.stop()

# ------------------------------------------------------------------------------------ analysis
with st.spinner("Analysing..."):
    analysis = det.analyze(suspect, sources)

scored = [s for s in analysis.sentences if s.analysable]
if not scored:
    st.warning("The suspect text has no sentences long enough to analyse (at least 4 words each).")
    st.stop()

coverage = det.coverage(analysis, "combined")
level, message = det.verdict(coverage)
flagged_n = sum(s.flagged for s in scored)

st.divider()
banner = {"High": st.error, "Partial": st.warning, "Low": st.success}[level]
banner(f"**{level} similarity.** {message} {coverage:.0%} of the suspect text ({flagged_n} of {len(scored)} sentences) matches the source.")

m1, m2, m3 = st.columns(3)
m1.metric("Text flagged (combined)", f"{coverage:.0%}")
m2.metric("Whole-document TF-IDF cosine", f"{analysis.doc_tfidf:.2f}")
m3.metric("Verbatim overlap (3-gram)", f"{analysis.doc_containment:.0%}")

st.subheader("Why the signals matter: each method on this pair")
rows = []
for m in METHODS:
    thr = det.thresholds[m]
    rows.append(
        {
            "Method": METHOD_LABELS[m],
            "Threshold": thr,
            "Sentences flagged": f"{sum(s.scores[m] >= thr for s in scored)} / {len(scored)}",
            "Share of text flagged": det.coverage(analysis, m),
        }
    )
st.dataframe(
    pd.DataFrame(rows),
    hide_index=True,
    width="stretch",
    column_config={
        "Threshold": st.column_config.NumberColumn(format="%.2f"),
        "Share of text flagged": st.column_config.ProgressColumn(format="percent", min_value=0.0, max_value=1.0),
    },
)
st.caption("Lexical methods (TF-IDF, 3-gram) drop when wording changes; the semantic score keeps tracking meaning.")

st.subheader("Highlighted matches")
legend = " ".join(
    f'<span style="background:{c};color:#111;padding:2px 8px;border-radius:3px;margin-right:6px">{l}</span>'
    for c, l in KIND_STYLE.values()
)
st.html(f"<div style='margin-bottom:10px'>{legend} <span style='opacity:.7'>Hover a highlight to see its match.</span></div>")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Suspect text**")
    st.html(highlight_suspect(suspect, analysis.sentences))
with c2:
    st.markdown("**Matched source passages**")
    st.html(highlight_source(sources, {s.source_text for s in scored if s.flagged}))

with st.expander("Sentence-by-sentence details", expanded=False):
    detail = pd.DataFrame(
        [
            {
                "#": s.idx + 1,
                "Suspect sentence": s.text,
                "Closest source sentence": s.source_text,
                "TF-IDF": s.components.get("tfidf", 0.0),
                "3-gram": s.components.get("containment", 0.0),
                "Semantic": s.components.get("semantic", 0.0),
                "Combined": s.scores["combined"],
                "Verdict": KIND_STYLE[s.kind][1] if s.flagged else "-",
            }
            for s in scored
        ]
    )
    score_fmt = st.column_config.NumberColumn(format="%.2f")
    st.dataframe(
        detail, hide_index=True, width="stretch",
        column_config={"TF-IDF": score_fmt, "3-gram": score_fmt, "Semantic": score_fmt, "Combined": score_fmt},
    )

st.caption(
    "Thresholds were set on a small synthetic benchmark (see `evaluate.py`), so treat a flag as evidence to review, not proof of plagiarism. "
    "A match can also be legitimate, for example a properly quoted and cited passage."
)
