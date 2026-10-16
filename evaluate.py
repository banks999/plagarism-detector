"""
evaluate.py - measure the detector on the synthetic benchmark in data.py.

    python evaluate.py                 # SBERT backend (needs `sentence-transformers`, internet on first run)
    python evaluate.py --model sentence-transformers/all-mpnet-base-v2

What it does
  1. Scores every suspect sentence against its source with all four methods.
  2. Picks each method's decision threshold by leave-one-topic-out cross-validation: the threshold used
     on a topic is tuned on the OTHER four topics only, so no number below is tuned on the data it is
     reported on.
  3. Reports sentence-level precision / recall / F1 (plus threshold-free ROC-AUC and average precision),
     document-level detection rate per plagiarism type, and the sentences the detector gets wrong.
  4. Tunes final thresholds on all data and saves them to thresholds.json for the app (SBERT backend only).

Everything is written to results/ (or --out).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import data
from engine import (
    DEFAULT_MODEL,
    DEFAULT_THRESHOLDS,
    METHODS,
    THRESHOLD_FLOORS,
    Detector,
    EmbedderUnavailable,
    load_embedder,
    save_thresholds,
)

DOC_COVERAGE = DEFAULT_THRESHOLDS["coverage"]  # a document is "flagged" if >= 30% of its words are in flagged sentences
LABELS = {"tfidf": "TF-IDF cosine", "containment": "3-gram containment", "semantic": "Semantic (embeddings)", "combined": "Combined"}


# ----------------------------------------------------------------------------- scoring
def score_cases(detector: Detector, cases) -> pd.DataFrame:
    rows = []
    for c in cases:
        analysis = detector.analyze(c.suspect_text, {"source": c.source_text})
        if len(analysis.sentences) != len(c.sentences):
            raise RuntimeError(f"sentence splitter disagrees with the benchmark on {c.id}")
        for s, label in zip(analysis.sentences, c.labels):
            rows.append(
                dict(
                    case=c.id, source_id=c.source_id, topic=c.topic, category=c.category,
                    sent_idx=s.idx, label=label, n_words=s.n_words, analysable=s.analysable,
                    **{m: s.scores[m] for m in METHODS},
                    suspect=s.text, matched=s.source_text,
                )
            )
    df = pd.DataFrame(rows)
    return df[df.analysable].reset_index(drop=True)


# ----------------------------------------------------------------------------- thresholds
def best_threshold(scores, labels) -> float:
    """Threshold that maximises F1 (midpoints between observed scores; middle of a tied plateau)."""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    u = np.unique(s)
    if len(u) == 1:
        return float(u[0])
    cands = np.concatenate([[u[0] - 1e-6], (u[:-1] + u[1:]) / 2, [u[-1] + 1e-6]])
    f1 = np.empty(len(cands))
    for k, t in enumerate(cands):
        p = s >= t
        tp = np.sum(p & (y == 1)); fp = np.sum(p & (y == 0)); fn = np.sum(~p & (y == 1))
        f1[k] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    best = np.flatnonzero(f1 >= f1.max() - 1e-12)
    return float(cands[best[len(best) // 2]])


def choose_threshold(scores, labels, method: str) -> float:
    """F1-optimal threshold, but never below the method's precision-first floor (see engine.THRESHOLD_FLOORS)."""
    return max(best_threshold(scores, labels), THRESHOLD_FLOORS[method])


def leave_one_topic_out(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean prediction per sentence and method; each topic is predicted with thresholds tuned on the others."""
    P = pd.DataFrame(False, index=df.index, columns=list(METHODS))
    for topic in df.topic.unique():
        test = df.topic == topic
        for m in METHODS:
            t = choose_threshold(df.loc[~test, m], df.loc[~test, "label"], m)
            P.loc[test, m] = df.loc[test, m] >= t
    return P


# ----------------------------------------------------------------------------- metrics
def sentence_metrics(df, P) -> pd.DataFrame:
    rows = []
    y = df.label.to_numpy()
    for m in METHODS:
        p = P[m].to_numpy()
        tp = int(np.sum(p & (y == 1))); fp = int(np.sum(p & (y == 0))); fn = int(np.sum(~p & (y == 1)))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append(
            dict(method=LABELS[m], precision=prec, recall=rec, f1=f1,
                 roc_auc=roc_auc_score(y, df[m]), avg_precision=average_precision_score(y, df[m]))
        )
    return pd.DataFrame(rows)


def sentence_rate_by_type(df, P) -> pd.DataFrame:
    """Share of sentences flagged, per kind of text. For the plagiarised types higher is better, for the rest lower."""
    group = df.category.where(df.category != "mixed", np.where(df.label == 1, "mixed: copied/paraphrased sentences",
                                                                  "mixed: original sentences"))
    order = ["exact", "synonym", "reorder", "paraphrase", "mixed: copied/paraphrased sentences",
             "mixed: original sentences", "hard_neg", "easy_neg"]
    rate = pd.DataFrame({m: P[m].groupby(group).mean() for m in METHODS}).reindex(order)
    return rate


def document_table(df, P):
    """Per document: word-weighted coverage per method, flagged if coverage >= DOC_COVERAGE."""
    rows = []
    for case, g in df.groupby("case", sort=False):
        rec = dict(case=case, category=g.category.iloc[0])
        for m in METHODS:
            rec[m] = g.n_words[P.loc[g.index, m]].sum() / g.n_words.sum()
        rows.append(rec)
    docs = pd.DataFrame(rows)
    flags = docs.copy()
    for m in METHODS:
        flags[m] = docs[m] >= DOC_COVERAGE
    rate = flags.groupby("category")[list(METHODS)].mean().reindex(list(data.CATEGORIES))
    return docs, rate


def document_summary(rate_flags: pd.DataFrame, docs: pd.DataFrame) -> pd.DataFrame:
    """Accuracy over all documents, treating the five plagiarism types as positive and both controls as negative."""
    pos = docs.category.isin(data.POSITIVE_CATEGORIES)
    rows = []
    for m in METHODS:
        flagged = docs[m] >= DOC_COVERAGE
        tp = int((flagged & pos).sum()); fp = int((flagged & ~pos).sum()); fn = int((~flagged & pos).sum())
        tn = int((~flagged & ~pos).sum())
        rows.append(dict(method=LABELS[m], detected=f"{tp}/{int(pos.sum())}", false_alarms=f"{fp}/{int((~pos).sum())}",
                         accuracy=(tp + tn) / len(docs)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- report helpers
def md_table(df: pd.DataFrame, fmt="{:.2f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = [fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r.tolist()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def error_examples(df, P, n=6):
    """Sentences worth discussing: lexical misses the combined score recovers, and where combined is wrong."""
    def block(title, sub, cols):
        out = [f"### {title}", ""]
        if sub.empty:
            return out + ["_none_", ""]
        for _, r in sub.head(n).iterrows():
            scores = ", ".join(f"{c}={r[c]:.2f}" for c in cols)
            out += [f"- **{r['category']}** ({scores})", f"  - suspect: {r['suspect']}", f"  - best source match: {r['matched']}"]
        return out + [""]

    cols = list(METHODS)
    rescued = df[(df.label == 1) & ~P.tfidf & ~P.containment & P.combined].sort_values("tfidf")
    fn = df[(df.label == 1) & ~P.combined].sort_values("combined")
    fp = df[(df.label == 0) & P.combined].sort_values("combined", ascending=False)
    return (
        block("Missed by both lexical methods, caught by the combined score", rescued, cols)
        + block("False negatives of the combined score (plagiarised, not flagged)", fn, cols)
        + block("False positives of the combined score (original text flagged)", fp, cols)
    )



# ----------------------------------------------------------------------------- chart
SERIES_COLORS = {"tfidf": "#2a78d6", "containment": "#eb6834", "semantic": "#1baf7a", "combined": "#eda100"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
GROUP_NAMES = {
    "exact": "Exact copy", "synonym": "Synonym\nswap", "reorder": "Sentences\nreordered", "paraphrase": "Paraphrase\n(hand-written)",
    "mixed: copied/paraphrased sentences": "Mixed doc:\nplagiarised\nsentences", "mixed: original sentences": "Mixed doc:\noriginal\nsentences",
    "hard_neg": "Same topic,\noriginal text", "easy_neg": "Other topic,\noriginal text",
}


def make_chart(rate: pd.DataFrame, path: Path, backend: str, standin: bool) -> bool:
    """Grouped bars: share of sentences flagged per kind of text and method. Returns False if matplotlib is missing."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        return False

    panels = [
        ("Plagiarised text: higher is better", ["exact", "synonym", "reorder", "paraphrase", "mixed: copied/paraphrased sentences"]),
        ("Original text: lower is better", ["mixed: original sentences", "hard_neg", "easy_neg"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True, facecolor=SURFACE, gridspec_kw={"width_ratios": [5, 3]})
    width = 0.2
    for ax, (title, cats) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        x = np.arange(len(cats))
        for k, m in enumerate(METHODS):
            vals = rate.loc[cats, m].to_numpy() * 100
            bars = ax.bar(x + (k - 1.5) * width, vals, width, color=SERIES_COLORS[m], edgecolor=SURFACE, linewidth=2, zorder=3)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=8, color=INK2)
        ax.set_xticks(x)
        ax.set_xticklabels([GROUP_NAMES[c] for c in cats], fontsize=9, color=INK2)
        ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.tick_params(axis="both", length=0, colors=INK2)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    axes[0].set_ylabel("% of sentences flagged", color=INK2, fontsize=10)
    axes[0].set_yticklabels(["0", "25", "50", "75", "100"], color=INK2)

    fig.legend(handles=[Patch(facecolor=SERIES_COLORS[m], label=LABELS[m]) for m in METHODS], loc="lower center", ncol=4,
               frameon=False, fontsize=10, labelcolor=INK2, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Share of sentences flagged, by kind of text and method", x=0.01, ha="left", fontsize=14, color=INK, y=0.99)
    note = f"Model: {backend}. Thresholds: cross-validated, precision-first floors. Synthetic benchmark, 10 source passages."
    if standin:
        note = "TEST STAND-IN, NOT SBERT: semantic and combined bars are not neural results.  " + note
    fig.text(0.01, 0.925, note, ha="left", fontsize=9, color="#d03b3b" if standin else INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 0.91))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return True

# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["auto", "sbert", "lsa"], default="auto",
                    help="'lsa' is an offline TEST stand-in, not a neural model; never report its numbers as SBERT")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="sentence-transformers model name")
    ap.add_argument("--out", default="results", help="output directory")
    ap.add_argument("--no-save-thresholds", action="store_true", help="do not write thresholds.json")
    args = ap.parse_args(argv)

    try:
        embedder = load_embedder(args.backend, args.model, lsa_corpus=data.all_sentences())
    except EmbedderUnavailable as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # use default thresholds while scoring; evaluation uses raw scores and tunes its own thresholds
    detector = Detector(embedder, thresholds=dict(DEFAULT_THRESHOLDS))
    cases = data.build_cases()
    print(f"Backend: {detector.backend} | {len(cases)} suspect documents | scoring...", flush=True)
    df = score_cases(detector, cases)

    P = leave_one_topic_out(df)
    sent = sentence_metrics(df, P)
    sent_rate = sentence_rate_by_type(df, P)
    docs, rate = document_table(df, P)
    doc_sum = document_summary(rate, docs)
    tuned = {m: best_threshold(df[m], df.label) for m in METHODS}
    final = {m: choose_threshold(df[m], df.label, m) for m in METHODS}
    final["coverage"] = DOC_COVERAGE

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["analysable"]).to_csv(out / "sentence_scores.csv", index=False)
    sent_rate.rename(columns=LABELS).to_csv(out / "sentence_flag_rate_by_type.csv")
    rate.rename(columns=LABELS).to_csv(out / "document_flag_rate_by_type.csv")

    rate_view = rate.rename(columns=LABELS).reset_index().rename(columns={"category": "document type"})
    sent_rate_view = sent_rate.rename(columns=LABELS).reset_index().rename(columns={"index": "kind of text", "category": "kind of text"})
    thr_view = pd.DataFrame(
        [dict(method=LABELS[m], f1_optimal=tuned[m], floor=THRESHOLD_FLOORS[m], used=final[m]) for m in METHODS]
    )
    standin = detector.backend.startswith("lsa")
    warning = (
        "> **WARNING: this run used the offline LSA stand-in, not a neural model. The 'semantic' and 'combined' columns "
        "are NOT SBERT results.**\n\n" if standin else ""
    )
    report = [
        "# Evaluation report", "", warning + f"Backend: `{detector.backend}`  |  documents: {len(docs)}  |  scored sentences: {len(df)}", "",
        "## How to read this",
        "- Data: 10 passages written for this project, with hand-written paraphrases (see `data.py`). It is a small, synthetic benchmark: "
        "it shows how the methods behave on each kind of plagiarism, not how they would score on real student essays.",
        "- Thresholds: for each topic they are tuned on the OTHER four topics (leave-one-topic-out), so nothing is tuned on the "
        "data it is reported on. The rule is threshold = max(F1-optimal cut-off, floor). The floors (see `engine.THRESHOLD_FLOORS`) "
        "are a precision-first design choice: this benchmark's negatives are easy, so the F1-optimal cut-offs alone can be low enough "
        "to flag too much on real text.",
        f"- A document counts as flagged when at least {DOC_COVERAGE:.0%} of its words sit in flagged sentences.", "",
        "## Sentence level (all scored sentences)", "", md_table(sent), "",
        "Precision / recall / F1 use the cross-validated thresholds above. ROC-AUC and average precision do not depend on any threshold.", "",
        "## Share of sentences flagged, by kind of text",
        "Plagiarised kinds (exact, synonym, reorder, paraphrase, mixed copied/paraphrased): higher is better. "
        "Original kinds (mixed original, hard_neg = same topic but original, easy_neg = other topic): lower is better.", "",
        md_table(sent_rate_view), "",
        "## Share of documents flagged, by type of document", "", md_table(rate_view), "",
        "## Document level: overall", "", md_table(doc_sum), "",
        "## Errors and examples", "", *error_examples(df, P), "",
        "## Final thresholds (tuned on all data, used by the app)", "", md_table(thr_view, "{:.3f}"), "",
        f"Document coverage threshold: {DOC_COVERAGE:.2f}",
    ]
    (out / "report.md").write_text("\n".join(report))

    print("\nSENTENCE LEVEL\n" + sent.round(3).to_string(index=False))
    print("\nSENTENCES FLAGGED BY KIND OF TEXT\n" + sent_rate_view.round(2).to_string(index=False))
    print("\nTHRESHOLDS USED\n" + thr_view.round(3).to_string(index=False))
    print("\nDOCUMENTS FLAGGED BY TYPE\n" + rate_view.round(2).to_string(index=False))
    print("\nDOCUMENT LEVEL OVERALL\n" + doc_sum.round(3).to_string(index=False))

    if make_chart(sent_rate, out / "flag_rate_by_type.png", detector.backend, standin):
        print(f"Chart saved to {out / 'flag_rate_by_type.png'}")
    else:
        print("(matplotlib not installed: skipped the chart. `pip install matplotlib` to get it.)")

    if standin:
        print("\nNOTE: lsa stand-in run; thresholds.json was NOT updated.")
    elif not args.no_save_thresholds:
        save_thresholds(detector.backend, final)
        print(f"\nSaved tuned thresholds for '{detector.backend}' to thresholds.json")
    print(f"Report and CSVs written to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
