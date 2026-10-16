"""
engine.py - text similarity and plagiarism detection engine.

For every sentence of a *suspect* text, the engine finds the most similar sentence in the
*source* text(s) using three signals, plus a weighted blend of them:

    tfidf        cosine similarity of TF-IDF vectors (stemmed unigrams + bigrams, stop-words removed)
    containment  share of the suspect sentence's word 3-grams that also occur in the source sentence
    semantic     cosine similarity of sentence-transformer (SBERT) embeddings
    combined     weighted blend of the three above

tfidf and containment are *lexical*: they catch copy-paste and light editing.
semantic is *neural*: it also catches paraphrasing, where the words change but the meaning stays.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from nltk.stem import PorterStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

METHODS = ("tfidf", "containment", "semantic", "combined")
MIN_WORDS = 4  # sentences shorter than this are not scored (headings, fragments)
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_WEIGHTS = {"semantic": 0.5, "tfidf": 0.3, "containment": 0.2}

# Precision-first policy: a sentence is never flagged on less evidence than this, even when the benchmark's
# (easy) negatives would allow a lower cut-off. A plagiarism accusation is costly, so false alarms matter.
# evaluate.py uses threshold = max(F1-optimal threshold, floor), both when it evaluates and when it saves.
THRESHOLD_FLOORS = {"tfidf": 0.25, "containment": 0.25, "semantic": 0.50, "combined": 0.35}
# Until evaluate.py has tuned thresholds for the active backend, the floors are used as they are.
DEFAULT_THRESHOLDS = {**THRESHOLD_FLOORS, "coverage": 0.30}  # coverage: share of the text that must be flagged
THRESHOLD_FILE = Path(__file__).with_name("thresholds.json")


# --------------------------------------------------------------------------------------
# Text processing
# --------------------------------------------------------------------------------------
_WORD = re.compile(r"[^\W_]+(?:['’]\w+)?")
_stem = lru_cache(maxsize=None)(PorterStemmer().stem)


def words(text: str) -> list[str]:
    """Lower-cased word tokens ("Earth's" -> "earth")."""
    return [re.split(r"['’]", w)[0] for w in _WORD.findall(text.lower())]


def content_stems(text: str) -> list[str]:
    """Stemmed, stop-word-free tokens."""
    return [_stem(w) for w in words(text) if w not in ENGLISH_STOP_WORDS]


def _tfidf_terms(text: str) -> list[str]:
    s = content_stems(text)
    return s + [f"{a}_{b}" for a, b in zip(s, s[1:])]


def shingles(text: str, n: int = 3) -> frozenset:
    """Set of word n-grams (unstemmed, stop-words kept, so verbatim copying shows up)."""
    w = words(text)
    if not w:
        return frozenset()
    if len(w) < n:
        return frozenset([tuple(w)])
    return frozenset(tuple(w[i : i + n]) for i in range(len(w) - n + 1))


@dataclass
class Span:
    text: str
    start: int
    end: int


_ABBREV = re.compile(r"\b(?:mr|mrs|ms|dr|prof|sr|jr|vs|etc|fig|no|st|inc|ltd|approx)\.$", re.I)
_BOUNDARY = re.compile(
    r'(?:(?<=[.!?])|(?<=[.!?]["”\')\]]))\s+(?=["“(\[]?[A-Z0-9])|\n\s*\n'
)


def split_sentences(text: str) -> list[Span]:
    """Rule-based sentence splitter that keeps character offsets into the original text."""
    pieces, pos = [], 0
    for m in _BOUNDARY.finditer(text):
        pieces.append((pos, m.start()))
        pos = m.end()
    pieces.append((pos, len(text)))

    spans: list[Span] = []
    for a, b in pieces:
        seg = text[a:b]
        stripped = seg.strip()
        if not stripped:
            continue
        start = a + (len(seg) - len(seg.lstrip()))
        end = start + len(stripped)
        if spans and _ABBREV.search(spans[-1].text):  # previous piece ended on "Dr." etc.: merge
            prev = spans.pop()
            spans.append(Span(text[prev.start : end], prev.start, end))
        else:
            spans.append(Span(stripped, start, end))
    return spans


# --------------------------------------------------------------------------------------
# Embedding backends
# --------------------------------------------------------------------------------------
class EmbedderUnavailable(RuntimeError):
    pass


class SBERTEmbedder:
    """Sentence-BERT embeddings. Uses the GPU automatically when CUDA is available."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.name = f"sbert:{model_name.split('/')[-1]}"

    def encode(self, sentences) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                list(sentences),
                batch_size=64,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )


class LSAEmbedder:
    """
    Offline stand-in (TF-IDF + truncated SVD). It exists ONLY so the pipeline can be tested on a
    machine that cannot download the neural model. It is much weaker than SBERT and its numbers
    must never be reported as semantic / SBERT results.
    """

    name = "lsa-standin"

    def __init__(self, corpus, dim: int = 64):
        self.vec = TfidfVectorizer(analyzer=content_stems, sublinear_tf=True)
        X = self.vec.fit_transform(list(corpus))
        k = max(2, min(dim, X.shape[1] - 1, X.shape[0] - 1))
        self.svd = TruncatedSVD(k, random_state=0).fit(X)

    def encode(self, sentences) -> np.ndarray:
        Z = self.svd.transform(self.vec.transform(list(sentences)))
        norms = np.linalg.norm(Z, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return Z / norms


def load_embedder(backend: str = "auto", model_name: str = DEFAULT_MODEL, lsa_corpus=None):
    if backend == "lsa":
        if lsa_corpus is None:
            raise ValueError("the lsa stand-in needs a corpus to fit on")
        return LSAEmbedder(lsa_corpus)
    try:
        return SBERTEmbedder(model_name)
    except Exception as exc:  # missing package, no internet for the first model download, ...
        raise EmbedderUnavailable(
            f"Could not load the SBERT model '{model_name}' ({type(exc).__name__}: {exc}).\n"
            "Fix: run `pip install sentence-transformers` and make sure you are online the first time "
            "(the ~90 MB model is downloaded once and then cached)."
        ) from exc


# --------------------------------------------------------------------------------------
# Thresholds
# --------------------------------------------------------------------------------------
def load_thresholds(backend_name: str, path: Path = THRESHOLD_FILE):
    """Tuned thresholds for this backend, or None if evaluate.py has not been run for it."""
    try:
        data = json.loads(Path(path).read_text())
        entry = data.get(backend_name)
        return {k: float(entry[k]) for k in DEFAULT_THRESHOLDS} if entry else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save_thresholds(backend_name: str, thresholds: dict, path: Path = THRESHOLD_FILE) -> None:
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        data = {}
    data[backend_name] = {k: round(float(thresholds[k]), 4) for k in DEFAULT_THRESHOLDS}
    Path(path).write_text(json.dumps(data, indent=2))


# --------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------
@dataclass
class SentenceResult:
    idx: int
    text: str
    start: int
    end: int
    n_words: int
    analysable: bool
    scores: dict = field(default_factory=dict)  # best score per method over ALL source sentences
    components: dict = field(default_factory=dict)  # tfidf/containment/semantic of the best combined match
    source_name: str = ""
    source_text: str = ""
    flagged: bool = False  # combined score >= combined threshold
    kind: str = "none"  # verbatim | edited | paraphrase | none


@dataclass
class Analysis:
    sentences: list
    n_source_sentences: int
    doc_tfidf: float  # cosine of whole-document TF-IDF vectors
    doc_containment: float  # share of suspect 3-grams found anywhere in the source(s)
    backend: str
    thresholds: dict
    thresholds_tuned: bool


def containment_matrix(susp_shingles, src_shingles) -> np.ndarray:
    """C[i, j] = |shingles(suspect_i) & shingles(source_j)| / |shingles(suspect_i)| (inverted index)."""
    index = defaultdict(list)
    for j, sh in enumerate(src_shingles):
        for g in sh:
            index[g].append(j)
    C = np.zeros((len(susp_shingles), len(src_shingles)))
    for i, sh in enumerate(susp_shingles):
        if not sh:
            continue
        hits = Counter()
        for g in sh:
            hits.update(index.get(g, ()))
        for j, c in hits.items():
            C[i, j] = c / len(sh)
    return C


def coverage_of(sentences, method: str, threshold: float) -> float:
    """Word-weighted share of the suspect text whose best match for `method` is >= threshold."""
    total = sum(s.n_words for s in sentences if s.analysable)
    if not total:
        return 0.0
    hit = sum(s.n_words for s in sentences if s.analysable and s.scores[method] >= threshold)
    return hit / total


# --------------------------------------------------------------------------------------
# Detector
# --------------------------------------------------------------------------------------
class Detector:
    def __init__(self, embedder, thresholds: dict | None = None, weights: dict | None = None, ngram: int = 3):
        self.embedder = embedder
        self.ngram = ngram
        w = dict(DEFAULT_WEIGHTS, **(weights or {}))
        total = sum(w.values())
        self.weights = {k: v / total for k, v in w.items()}
        tuned = thresholds or load_thresholds(embedder.name)
        self.thresholds_tuned = tuned is not None
        self.thresholds = dict(DEFAULT_THRESHOLDS, **(tuned or {}))

    @property
    def backend(self) -> str:
        return self.embedder.name

    @staticmethod
    def _tfidf_matrix(a: list[str], b: list[str]) -> np.ndarray:
        try:
            X = TfidfVectorizer(analyzer=_tfidf_terms, sublinear_tf=True).fit_transform(a + b)
        except ValueError:  # nothing but stop-words
            return np.zeros((len(a), len(b)))
        return (X[: len(a)] @ X[len(a) :].T).toarray()

    def analyze(self, suspect: str, sources) -> Analysis:
        if isinstance(sources, str):
            sources = {"source": sources}

        susp_spans = split_sentences(suspect)
        src = [(name, sp) for name, txt in sources.items() for sp in split_sentences(txt)]
        n, m = len(susp_spans), len(src)

        sentences = [
            SentenceResult(
                idx=i,
                text=sp.text,
                start=sp.start,
                end=sp.end,
                n_words=len(words(sp.text)),
                analysable=len(words(sp.text)) >= MIN_WORDS,
                scores={k: 0.0 for k in METHODS},
            )
            for i, sp in enumerate(susp_spans)
        ]
        if n == 0 or m == 0:
            return Analysis(sentences, m, 0.0, 0.0, self.backend, self.thresholds, self.thresholds_tuned)

        susp_txt = [s.text for s in susp_spans]
        src_txt = [sp.text for _, sp in src]

        T = self._tfidf_matrix(susp_txt, src_txt)
        C = containment_matrix(
            [shingles(t, self.ngram) for t in susp_txt], [shingles(t, self.ngram) for t in src_txt]
        )
        S = np.clip(self.embedder.encode(susp_txt) @ self.embedder.encode(src_txt).T, 0.0, 1.0)

        usable_src = np.array([len(words(t)) >= MIN_WORDS for t in src_txt])
        for mat in (T, C, S):
            mat[:, ~usable_src] = 0.0
        w = self.weights
        M = w["tfidf"] * T + w["containment"] * C + w["semantic"] * S
        mats = {"tfidf": T, "containment": C, "semantic": S, "combined": M}

        thr = self.thresholds
        for i, res in enumerate(sentences):
            if not res.analysable:
                continue
            res.scores = {k: float(mats[k][i].max()) for k in METHODS}
            j = int(M[i].argmax())
            res.components = {"tfidf": float(T[i, j]), "containment": float(C[i, j]), "semantic": float(S[i, j])}
            res.source_name, res.source_text = src[j][0], src_txt[j]
            res.flagged = res.scores["combined"] >= thr["combined"]
            if res.flagged:
                comp = res.components
                if comp["containment"] >= 0.6:
                    res.kind = "verbatim"
                elif comp["tfidf"] >= 0.5 or comp["containment"] >= 0.25:
                    res.kind = "edited"
                else:
                    res.kind = "paraphrase"

        # document-level lexical measures
        all_src = " ".join(src_txt)
        try:
            X = TfidfVectorizer(analyzer=_tfidf_terms, sublinear_tf=True).fit_transform([suspect, all_src])
            doc_tfidf = float((X[0] @ X[1].T).toarray()[0, 0])
        except ValueError:
            doc_tfidf = 0.0
        sh_susp, sh_src = shingles(suspect, self.ngram), shingles(all_src, self.ngram)
        doc_cont = len(sh_susp & sh_src) / len(sh_susp) if sh_susp else 0.0

        return Analysis(sentences, m, doc_tfidf, doc_cont, self.backend, self.thresholds, self.thresholds_tuned)

    def coverage(self, analysis: Analysis, method: str = "combined", threshold: float | None = None) -> float:
        thr = self.thresholds[method] if threshold is None else threshold
        return coverage_of(analysis.sentences, method, thr)

    def verdict(self, coverage: float) -> tuple[str, str]:
        if coverage >= 0.6:
            return "High", "Most of the text matches the source."
        if coverage >= self.thresholds["coverage"]:
            return "Partial", "Parts of the text match the source."
        return "Low", "No substantial match with the source."
