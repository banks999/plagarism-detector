# Evaluation report

Backend: `sbert:all-MiniLM-L6-v2`  |  documents: 70  |  scored sentences: 350

## How to read this
- Data: 10 passages written for this project, with hand-written paraphrases (see `data.py`). It is a small, synthetic benchmark: it shows how the methods behave on each kind of plagiarism, not how they would score on real student essays.
- Thresholds: for each topic they are tuned on the OTHER four topics (leave-one-topic-out), so nothing is tuned on the data it is reported on. The rule is threshold = max(F1-optimal cut-off, floor). The floors (see `engine.THRESHOLD_FLOORS`) are a precision-first design choice: this benchmark's negatives are easy, so the F1-optimal cut-offs alone can be low enough to flag too much on real text.
- A document counts as flagged when at least 30% of its words sit in flagged sentences.

## Sentence level (all scored sentences)

| method | precision | recall | f1 | roc_auc | avg_precision |
|---|---|---|---|---|---|
| TF-IDF cosine | 1.00 | 0.81 | 0.89 | 0.98 | 0.99 |
| 3-gram containment | 1.00 | 0.72 | 0.84 | 0.93 | 0.95 |
| Semantic (embeddings) | 0.98 | 1.00 | 0.99 | 1.00 | 1.00 |
| Combined | 1.00 | 0.97 | 0.99 | 1.00 | 1.00 |

Precision / recall / F1 use the cross-validated thresholds above. ROC-AUC and average precision do not depend on any threshold.

## Share of sentences flagged, by kind of text
Plagiarised kinds (exact, synonym, reorder, paraphrase, mixed copied/paraphrased): higher is better. Original kinds (mixed original, hard_neg = same topic but original, easy_neg = other topic): lower is better.

| kind of text | TF-IDF cosine | 3-gram containment | Semantic (embeddings) | Combined |
|---|---|---|---|---|
| exact | 1.00 | 1.00 | 1.00 | 1.00 |
| synonym | 0.96 | 0.92 | 1.00 | 1.00 |
| reorder | 1.00 | 1.00 | 1.00 | 1.00 |
| paraphrase | 0.30 | 0.04 | 0.98 | 0.90 |
| mixed: copied/paraphrased sentences | 0.75 | 0.55 | 1.00 | 0.95 |
| mixed: original sentences | 0.00 | 0.00 | 0.00 | 0.00 |
| hard_neg | 0.00 | 0.00 | 0.08 | 0.00 |
| easy_neg | 0.00 | 0.00 | 0.00 | 0.00 |

## Share of documents flagged, by type of document

| document type | TF-IDF cosine | 3-gram containment | Semantic (embeddings) | Combined |
|---|---|---|---|---|
| exact | 1.00 | 1.00 | 1.00 | 1.00 |
| synonym | 1.00 | 1.00 | 1.00 | 1.00 |
| reorder | 1.00 | 1.00 | 1.00 | 1.00 |
| paraphrase | 0.40 | 0.00 | 1.00 | 1.00 |
| mixed | 0.50 | 0.10 | 1.00 | 0.90 |
| hard_neg | 0.00 | 0.00 | 0.20 | 0.00 |
| easy_neg | 0.00 | 0.00 | 0.00 | 0.00 |

## Document level: overall

| method | detected | false_alarms | accuracy |
|---|---|---|---|
| TF-IDF cosine | 39/50 | 0/20 | 0.84 |
| 3-gram containment | 31/50 | 0/20 | 0.73 |
| Semantic (embeddings) | 50/50 | 2/20 | 0.97 |
| Combined | 49/50 | 0/20 | 0.99 |

## Errors and examples

### Missed by both lexical methods, caught by the combined score

- **paraphrase** (tfidf=0.03, containment=0.00, semantic=0.77, combined=0.39)
  - suspect: As people abandoned agricultural work in search of employment, towns expanded quickly, and many residents endured overcrowded, unhealthy housing.
  - best source match: Cities grew rapidly as workers left farms to look for jobs, often living in crowded and unhealthy conditions.
- **paraphrase** (tfidf=0.04, containment=0.00, semantic=0.74, combined=0.38)
  - suspect: Alterations to these flows can influence weather, fishing grounds and where sea creatures live.
  - best source match: Changes in these currents can affect weather patterns, fisheries, and the distribution of marine life.
- **mixed** (tfidf=0.06, containment=0.00, semantic=0.72, combined=0.36)
  - suspect: When buyers' appetite jumps unexpectedly, goods become scarce and prices usually go up until output rises to meet it.
  - best source match: If demand suddenly increases, shortages appear and prices tend to climb until supply catches up.
- **paraphrase** (tfidf=0.06, containment=0.00, semantic=0.72, combined=0.36)
  - suspect: When buyers' appetite jumps unexpectedly, goods become scarce and prices usually go up until output rises to meet it.
  - best source match: If demand suddenly increases, shortages appear and prices tend to climb until supply catches up.
- **mixed** (tfidf=0.06, containment=0.00, semantic=0.81, combined=0.43)
  - suspect: Since industrialisation, the combustion of coal, oil and gas along with deforestation has raised the atmospheric levels of those gases.
  - best source match: Burning fossil fuels and clearing forests have increased the concentration of these gases in the atmosphere since the industrial era.
- **paraphrase** (tfidf=0.06, containment=0.00, semantic=0.85, combined=0.44)
  - suspect: Reading material grew more affordable, literacy gradually climbed, and fresh ideas moved around Europe more quickly than at any previous point.
  - best source match: Books became cheaper, literacy rates slowly rose, and new ideas travelled across the continent faster than ever before.

### False negatives of the combined score (plagiarised, not flagged)

- **paraphrase** (tfidf=0.04, containment=0.00, semantic=0.59, combined=0.30)
  - suspect: Since rewriting a past block would invalidate all the fingerprints after it, any attempt to falsify the record can be spotted readily.
  - best source match: Because changing an old block would alter every hash that follows, tampering with the history is easy to detect.
- **paraphrase** (tfidf=0.00, containment=0.00, semantic=0.60, combined=0.30)
  - suspect: When firms make more than customers are looking for, unsold stock accumulates and vendors have to lower what they charge.
  - best source match: If suppliers produce more than buyers want, surpluses build up and sellers are forced to cut prices.
- **paraphrase** (tfidf=0.08, containment=0.00, semantic=0.56, combined=0.30)
  - suspect: Entries are bundled into units called blocks, with every unit storing the cryptographic fingerprint of its predecessor.
  - best source match: Transactions are grouped into blocks, and each block contains a cryptographic hash of the block that came before it.
- **mixed** (tfidf=0.03, containment=0.00, semantic=0.62, combined=0.32)
  - suspect: In the depths, chilly and briny water grows heavy and descends, then drifts around the world for centuries.
  - best source match: Deep currents form when cold, salty water becomes dense and sinks, slowly circulating around the globe over hundreds of years.
- **paraphrase** (tfidf=0.03, containment=0.00, semantic=0.62, combined=0.32)
  - suspect: In the depths, chilly and briny water grows heavy and descends, then drifts around the world for centuries.
  - best source match: Deep currents form when cold, salty water becomes dense and sinks, slowly circulating around the globe over hundreds of years.
- **paraphrase** (tfidf=0.07, containment=0.00, semantic=0.63, combined=0.33)
  - suspect: Nearly every living thing relies on it, because it provides the foundation of food webs as well as the oxygen we breathe.
  - best source match: Almost all life on Earth depends on this process, since it supplies both the food chain and the oxygen in the atmosphere.

### False positives of the combined score (original text flagged)

_none_


## Final thresholds (tuned on all data, used by the app)

| method | f1_optimal | floor | used |
|---|---|---|---|
| TF-IDF cosine | 0.056 | 0.250 | 0.250 |
| 3-gram containment | 0.020 | 0.250 | 0.250 |
| Semantic (embeddings) | 0.578 | 0.500 | 0.578 |
| Combined | 0.303 | 0.350 | 0.350 |

Document coverage threshold: 0.30