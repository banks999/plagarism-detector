# Text Similarity and Plagiarism Detection Tool

Compares a suspect text with one or more source texts and highlights which sentences were copied, lightly edited, or
paraphrased. Lexical methods (TF-IDF, n-gram overlap) find copying; sentence embeddings (SBERT) also find paraphrasing.

## Run it (about 5 minutes)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python evaluate.py        # 1. benchmark + tune thresholds (downloads the ~90 MB SBERT model the first time)
streamlit run app.py      # 2. the demo app (picks up the tuned thresholds automatically)
```

Do step 1 once while you have internet, before the presentation, so the model is cached and works offline afterwards.
GPU is optional: if PyTorch sees CUDA it is used automatically, otherwise it runs on CPU, which is fine for this model size.

## What each file does

| File | Role |
|---|---|
| `engine.py` | The detector: sentence splitting, TF-IDF, 3-gram containment, SBERT, combined score, sentence matching |
| `data.py` | Synthetic benchmark: 10 passages, hand-written paraphrases, generated copy / synonym / reorder / mixed / control documents |
| `evaluate.py` | Scores the benchmark, tunes thresholds, writes `results/report.md`, CSVs and `results/flag_rate_by_type.png` |
| `app.py` | Streamlit demo with demo buttons, highlighted matches and a per-method comparison |
| `thresholds.json` | Created by `evaluate.py`; the app loads it |

## How it works

For every suspect sentence the engine finds its best match among the source sentences, using:

| Signal | What it measures | Good at | Weak at |
|---|---|---|---|
| TF-IDF cosine | Shared wording, weighted by rarity (stemmed unigrams and bigrams, stop-words removed) | Copying, light edits | Paraphrase with different vocabulary |
| 3-gram containment | Share of the sentence's word triples found in a source sentence | Verbatim copying | Almost any rewording |
| Semantic (SBERT `all-MiniLM-L6-v2`) | Cosine of sentence embeddings | Paraphrase | Related but different sentences can score fairly high |
| Combined | 0.5 semantic + 0.3 TF-IDF + 0.2 containment | Balance | Weights are a design choice, not learned |

A document's headline number is the share of its words that sit in flagged sentences (coverage). Sentences are labelled
verbatim / edited / paraphrase from which signals fired; that label is for display and is not evaluated.

## Evaluation design (say this in the presentation)

- Data: 10 passages written for this project, five topics with two passages each. Each source yields seven suspect documents:
  exact, synonym swap, reordered, hand-written paraphrase, mixed (copied + paraphrased + unrelated sentences), and two
  controls: same topic but original (`hard_neg`) and a different topic (`easy_neg`). Every sentence has a known label.
- Thresholds are chosen by leave-one-topic-out cross-validation: the cut-offs applied to a topic are tuned on the other four.
- Rule: threshold = max(F1-optimal cut-off, floor). The floors in `engine.THRESHOLD_FLOORS` are a precision-first design
  decision, because the benchmark's negatives are easy and F1-optimal cut-offs alone would be low enough to flag too much
  real text. `results/report.md` lists the tuned and the floored values side by side.
- `ROC-AUC` and `average precision` in the report do not depend on thresholds at all.

## Limitations (be upfront about these)

- The benchmark is small and synthetic; results show how the methods behave per type of plagiarism, not how accurate the tool
  is on real student essays. The synonym swaps are automatic, the paraphrases are hand-written by the project author.
- The demo buttons use passages from the benchmark. Also test with text you write yourself.
- English only; `.txt` / `.md` upload only; no web search (it compares against the sources you give it).
- A flag is evidence to review, not proof: properly quoted and cited text will match too.
- Thresholds and the combined weights are design choices tuned on a small data set.

## Troubleshooting

- `Could not load the SBERT model`: you were offline on the first run. Connect once and rerun `python evaluate.py`.
- Port busy: `streamlit run app.py --server.port 8502`.
- Check for an NVIDIA GPU: `nvidia-smi` (optional; CPU works).
