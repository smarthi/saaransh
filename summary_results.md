# saaransh gap-filling benchmark — results

ColQwen2 MaxSim (exact ceiling) vs MUVERA (single-vector FDE) vs Gemma 4 12B pooled, on 4 ViDoRe datasets (docvqa, arxivqa, infovqa, tabfquad; 200 docs / 200 queries each).

## MUVERA vs exact-MaxSim ceiling

| dataset | ceiling nDCG@5 | best MUVERA nDCG@5 | % of ceiling | best config | storage/page | ceiling-lost-by-MUVERA |
|---|---|---|---|---|---|---|
| arxivqa | 0.895 | 0.830 | 93% | muvera[DE\|k8\|r8\|c-] | 1024 KB | 21/200 lost |
| docvqa | 0.655 | 0.534 | 82% | muvera[CA\|k8\|r8\|c-] | 1024 KB | 29/200 lost |
| infovqa | 0.914 | 0.839 | 92% | muvera[CA\|k8\|r8\|c-] | 1024 KB | 23/200 lost |
| tabfquad | 0.594 | 0.557 | 94% | muvera[DE\|k8\|r4\|c-] | 512 KB | 35/200 lost |

(`ceiling-lost-by-MUVERA` updated from the original run: arxivqa and tabfquad's failure-analysis step had incorrectly used `calibrated_eigenbasis` for all four datasets instead of each dataset's actual best config from `muvera_sweep.csv`. Re-run below with the correct per-dataset config: arxivqa/tabfquad use `default_identity`, not `calibrated_eigenbasis`.)

## Gemma 4 12B pooled — precision sweep (docvqa, 200/200)

MLX q8/q4 not run — hidden-state extraction for that backend is an unimplemented stub in the code (`gemma4_pooled.py`), not something evaluated here.

```
precision,recall@1,ndcg@5,bytes_per_doc
bf16,0.0100,0.0248,15360
```

```
pipeline                                  index    dim    B/doc  recall@1  recall@5    ndcg@5    mrr@10   idx s   qry s
gemma4-12b[transformers/bf16/mean]        numpy   3840    15360     0.010     0.040     0.025     0.022   277.1    56.4
```

## Failure analysis — per dataset (bucketing corrected: structural features, not first-word)

Bucketing was corrected from the original first-word heuristic (which only recognized English
question words and degenerated to a single "other" bucket on the French tabfquad queries) to a
structural composite: `has_digit`, `has_at`, `has_color`, `has_quoted_or_acronym`,
`has_identifier_noun` → `specific_entity_lookup` vs `descriptive_general`. Each dataset below
re-run with its own actual best MUVERA config from `muvera_sweep.csv` (not a single config
assumed across all four).

`specific_entity_lookup` avg rank_delta vs `descriptive_general`, across all four:

| dataset | specific_entity_lookup | descriptive_general | diff |
|---|---|---|---|
| docvqa | +1.29 (n=55) | +1.19 (n=145) | +0.10 |
| arxivqa | +0.78 (n=96) | +0.47 (n=104) | +0.31 |
| infovqa | +0.58 (n=81) | +0.54 (n=119) | +0.04 |
| tabfquad | +0.81 (n=126) | +0.18 (n=74) | +0.63 |

`specific_entity_lookup` is higher than `descriptive_general` in all four datasets, though the
gap ranges from small (docvqa, infovqa) to substantial (arxivqa, tabfquad). Full per-dataset
console output below.

### docvqa (calibrated_eigenbasis, k=8, r=8)

```
cache: 200 docs, 200 queries, token_dim=128
wrote ./results/docvqa/failure_report.csv  (200 queries)

ceiling got it right but MUVERA lost it: 29/200 queries

avg rank_delta: specific-entity-lookup vs descriptive-general (higher = MUVERA relatively worse):
  specific_entity_lookup n=55   avg_delta=+1.29
  descriptive_general    n=145  avg_delta=+1.19

same split, by individual structural feature (not mutually exclusive):
  has_digit              n=26   avg_delta=+1.19   (without: n=174  avg_delta=+1.22)
  has_color              n=2    avg_delta=+5.00   (without: n=198  avg_delta=+1.18)
  has_quoted_or_acronym  n=27   avg_delta=+1.26   (without: n=173  avg_delta=+1.21)
  has_identifier_noun    n=4    avg_delta=+1.50   (without: n=196  avg_delta=+1.21)

corr(query length, rank_delta) = -0.144

worst 20 regressions (ceiling rank vs MUVERA rank):
  delta= 10  ceiling=   1  muvera=None  "To whom is the letter is addressed?"
  delta= 10  ceiling=   1  muvera=None  "What is the Manuscript number specified in the 'Title' ?"
  delta= 10  ceiling=   1  muvera=None  "what is written at the middle of round seal?"
  delta= 10  ceiling=   1  muvera=None  "what is re written in place of 34.8gallons"
  delta= 10  ceiling=   1  muvera=None  "What is the priority of the Article WMC(2)?"
  delta= 10  ceiling=   1  muvera=None  "What does this document relate to ?"
  delta= 10  ceiling=   1  muvera=None  "What is the heading printed in red?"
  delta=  9  ceiling=   2  muvera=None  "Which year was the meeting held?"
  delta=  9  ceiling=   2  muvera=None  "Who is the Medical Monitor or designee?"
  delta=  9  ceiling=   2  muvera=None  "What is the Page Number?"
  delta=  9  ceiling=   2  muvera=None  "What is the table number?"
  delta=  9  ceiling=   2  muvera=None  "What is the name of the person who has signed the letter?"
  delta=  8  ceiling=   3  muvera=None  "What is the table number?"
  delta=  8  ceiling=   1  muvera=   9  "what is the trend of patient growth?"
  delta=  7  ceiling=   2  muvera=   9  "What is the name of the person in the CC field ?"
  delta=  7  ceiling=   1  muvera=   8  "Which university 'letterhead' is given?"
  delta=  7  ceiling=   4  muvera=None  "When was this meeting happened?"
  delta=  7  ceiling=   1  muvera=   8  "What is the team leader's name?"
  delta=  6  ceiling=   5  muvera=None  "What is the table number?"
  delta=  6  ceiling=   4  muvera=  10  "What is the page number?"
```

### arxivqa (default_identity, k=8, r=8)

```
cache: 200 docs, 200 queries, token_dim=128
wrote ./results/arxivqa/failure_report.csv  (200 queries)

ceiling got it right but MUVERA lost it: 21/200 queries

avg rank_delta: specific-entity-lookup vs descriptive-general (higher = MUVERA relatively worse):
  specific_entity_lookup n=96   avg_delta=+0.78
  descriptive_general    n=104  avg_delta=+0.47

same split, by individual structural feature (not mutually exclusive):
  has_digit              n=48   avg_delta=+0.25   (without: n=152  avg_delta=+0.74)
  has_color              n=17   avg_delta=+2.94   (without: n=183  avg_delta=+0.40)
  has_quoted_or_acronym  n=43   avg_delta=+0.30   (without: n=157  avg_delta=+0.71)

corr(query length, rank_delta) = -0.080

worst 20 regressions (ceiling rank vs MUVERA rank):
  delta= 10  ceiling=   1  muvera=None  "Based on the vector fields shown in the bottom portion of the figure, what characteristic changes from panel A to panel D?"
  delta= 10  ceiling=   1  muvera=None  "What does the graph suggest about the scaling of different operators in the large-N limit?"
  delta= 10  ceiling=   1  muvera=None  "What interaction is represented by the solid blue line in Figure (a)?"
  delta=  9  ceiling=   2  muvera=None  "What does the solid red line represent in the figure?"
  delta=  9  ceiling=   2  muvera=None  "Based on the geometry shown in figure (a), what does the blue rhombus most likely represent?"
  delta=  8  ceiling=   3  muvera=None  "What feature of the graph allows for the comparison of different \(\eta\) values?"
  delta=  8  ceiling=   3  muvera=None  "At what approximate x-axis value does the purple diamond dataset begin to level off?"
  delta=  8  ceiling=   1  muvera=   9  "According to the diagram, which component is NOT part of the scalable and efficient execution with Region Templates?"
  delta=  8  ceiling=   3  muvera=None  "Based on the figure, what can be inferred about the location of the observed subject?"
  delta=  8  ceiling=   2  muvera=  10  "What does panel (a) of the figure primarily illustrate?"
  delta=  7  ceiling=   1  muvera=   8  "If the colors in the image represent different categories, what can be inferred about the category represented by the color cyan?"
  delta=  7  ceiling=   2  muvera=   9  "Based on the figure, which statement is true regarding the direction of wave propagation?"
  delta=  5  ceiling=   1  muvera=   6  "Based on the figure, how does changing the parameter V from 0.1 to 0.2 affect the system's transition to disorder for N=200?"
  delta=  5  ceiling=   6  muvera=None  "What does the dashed line represent in this layered network diagram?"
  delta=  5  ceiling=   6  muvera=None  "What is the significance of the two different sizes mentioned in figure (c)?"
  delta=  4  ceiling=   1  muvera=   5  "What is the relationship between the elements labeled 'x' and 'y' on the left side of the figure and the elements 'a', 'b', 'c' on the right side of the figure?"
  delta=  4  ceiling=   1  muvera=   5  "What might the purpose of the white silhouette against the black background be?"
  delta=  4  ceiling=   7  muvera=None  "What does the color scale in panel (b) represent?"
  delta=  3  ceiling=   1  muvera=   4  "Based on the phase space plots in figure (b), what can be inferred about the state of the system at point 2?"
  delta=  2  ceiling=   2  muvera=   4  "What physical phenomenon could the pattern of lines on the right side of the figure represent?"
```

### infovqa (calibrated_eigenbasis, k=8, r=8)

```
cache: 200 docs, 200 queries, token_dim=128
wrote ./results/infovqa/failure_report.csv  (200 queries)

ceiling got it right but MUVERA lost it: 23/200 queries

avg rank_delta: specific-entity-lookup vs descriptive-general (higher = MUVERA relatively worse):
  specific_entity_lookup n=81   avg_delta=+0.58
  descriptive_general    n=119  avg_delta=+0.54

same split, by individual structural feature (not mutually exclusive):
  has_digit              n=49   avg_delta=+0.16   (without: n=151  avg_delta=+0.68)
  has_color              n=4    avg_delta=+1.25   (without: n=196  avg_delta=+0.54)
  has_quoted_or_acronym  n=31   avg_delta=+0.58   (without: n=169  avg_delta=+0.55)
  has_identifier_noun    n=5    avg_delta=+3.80   (without: n=195  avg_delta=+0.47)

corr(query length, rank_delta) = +0.017

worst 20 regressions (ceiling rank vs MUVERA rank):
  delta= 10  ceiling=   1  muvera=None  "Who is the CEO of JP Morgan?"
  delta= 10  ceiling=   1  muvera=None  "which area is represented by a smiling face icon?"
  delta= 10  ceiling=   1  muvera=None  "what is the combined total economy of UK and Germany in trillion dollars?"
  delta= 10  ceiling=   1  muvera=None  "What is the Twitter handle of the designer of infographic?"
  delta=  9  ceiling=   2  muvera=None  "Which social media referred more business than LinkedIn, YouTube, and Google +?"
  delta=  8  ceiling=   3  muvera=None  "what is the colour that comes in between green and orange, yellow or red"
  delta=  8  ceiling=   3  muvera=None  "What is the email address provided?"
  delta=  7  ceiling=   4  muvera=None  "Which is the most popular social media platform?"
  delta=  6  ceiling=   1  muvera=   7  "Which social networking website is used  for personal use by most of the respondents according to the survey - Twitter, Skype, LinkedIn, Facebook?"
  delta=  6  ceiling=   1  muvera=   7  "What % of girls say that they never feel comfortable using school latrines"
  delta=  5  ceiling=   4  muvera=   9  "The highest number of soldiers died in which war?"
  delta=  5  ceiling=   1  muvera=   6  "What are the three forms of violence against women?"
  delta=  4  ceiling=   1  muvera=   5  "Who is the author of Cat's Cradle"
  delta=  4  ceiling=   1  muvera=   5  "What is the difference in the average time spent on radio in 2018 and 2019?"
  delta=  3  ceiling=   3  muvera=   6  "What percent of Pinterest users are male?"
  delta=  3  ceiling=   1  muvera=   4  "What is the distance (in light-years) of Vega from earth?"
  delta=  3  ceiling=   2  muvera=   5  "What are the types of abuses other than economic & psychological?"
  delta=  2  ceiling=   1  muvera=   3  "Which category of apps enjoy the least percentage of popularity?"
  delta=  2  ceiling=   1  muvera=   3  "How many teams are there in the Masters Champions League?"
  delta=  2  ceiling=   1  muvera=   3  "how many boys are there in the world for 100 girls?"
```

### tabfquad (default_identity, k=8, r=4)

```
cache: 200 docs, 200 queries, token_dim=128
wrote ./results/tabfquad/failure_report.csv  (200 queries)

ceiling got it right but MUVERA lost it: 35/200 queries

avg rank_delta: specific-entity-lookup vs descriptive-general (higher = MUVERA relatively worse):
  specific_entity_lookup n=126  avg_delta=+0.81
  descriptive_general    n=74   avg_delta=+0.18

same split, by individual structural feature (not mutually exclusive):
  has_digit              n=104  avg_delta=+0.82   (without: n=96   avg_delta=+0.31)
  has_quoted_or_acronym  n=47   avg_delta=+0.55   (without: n=153  avg_delta=+0.58)
  has_identifier_noun    n=2    avg_delta=+0.50   (without: n=198  avg_delta=+0.58)

corr(query length, rank_delta) = -0.117

worst 20 regressions (ceiling rank vs MUVERA rank):
  delta= 10  ceiling=   1  muvera=None  "De combien étaient les ventes en millions d'euros en 2011 ?"
  delta= 10  ceiling=   1  muvera=None  "Quelles ont été les variations du chiffre d'affaires des entreprises aéronautiques entre 2009 et 2011 ?"
  delta= 10  ceiling=   1  muvera=None  "Quel est l'ordre de grandeur de l'indemnité de stage dans les coûts de fonctionnement d'un projet universitaire ?"
  delta= 10  ceiling=   1  muvera=None  "quel est le total de passifs au 31 decembre 2011 ?"
  delta= 10  ceiling=   1  muvera=None  "Quelle était la valeur des actifs non courants d'Air France en 2012 ?"
  delta= 10  ceiling=   1  muvera=None  "Peut-on trouver des exemples de structures de primes basées sur l'égalité ou la performance dans les grandes entreprises ?"
  delta=  9  ceiling=   2  muvera=None  "Combien d'actions ont été vendues entre décembre 2010 et mars 2011 ?"
  delta=  9  ceiling=   2  muvera=None  "Quelles étaient les principales catégories de dette financière d'une entreprise à la fin de l'année 2019 et 2020?"
  delta=  8  ceiling=   3  muvera=None  "Quel était le résultat avant impots des caisses régionales en 2016 ?"
  delta=  7  ceiling=   4  muvera=None  "Quelles étaient les fourchettes de pourcentage pour les années 2006 à 2008 concernant un indicateur économique par zone géographique?"
  delta=  7  ceiling=   2  muvera=   9  "Quel était le montant total des litiges en euros enregistré en juin 2017?"
  delta=  7  ceiling=   1  muvera=   8  "Quels sont les tarifs moyens pratiqués par les sociétés de transport pour différentes classes de véhicules?"
  delta=  6  ceiling=   4  muvera=  10  "Quel est le rôle des régions dans le financement des services d'incendie et de secours comparé à d'autres échelons locaux ?"
  delta=  6  ceiling=   4  muvera=  10  "Sur l'année 2009-2010, de combien de % ont augmenté les droits de vol ?"
  delta=  5  ceiling=   1  muvera=   6  "Comment se répartit la durée d'occupation des logements entre locataires et propriétaires?"
  delta=  4  ceiling=   7  muvera=None  "Comment a évolué le résultat net des sociétés consolidées dans le secteur aérien entre 2011 et 2012?"
  delta=  4  ceiling=   2  muvera=   6  "Quelles sont les différentes méthodes de calcul des primes dans les entreprises françaises ?"
  delta=  4  ceiling=   2  muvera=   6  "Quels sont les montants des obligations locatives de crédit-bail enregistrés en mars et septembre 2006 ?"
  delta=  4  ceiling=   6  muvera=  10  "Quelles conséquences a une variation de la marge opérationnelle sur les projections financières d'une société?"
  delta=  4  ceiling=   7  muvera=None  "Quels sont les principaux postes de dépenses externes pour les compagnies aériennes?"
```

## Run notes (fixes applied to get here, not results)

- `scripts/failure_analysis.py` was misplaced at `src/saaransh/failure_analysis.py`; moved to match `run.sh` and its own docstring.
- `run.sh`'s `COLQWEN_MODEL`/`GEMMA_MODEL` pointed at nonexistent local paths; fixed to use real HF repo ids + `--cache-dir`.
- Full pytest suite (20 passed, 1 skipped — unrelated missing `pdf` extra) required `KMP_DUPLICATE_LIB_OK=TRUE` to avoid a faiss+torch duplicate-OpenMP crash on macOS.
- Gemma precision sweep restricted to `bf16`; MLX q8/q4 hidden-state extraction is an unimplemented stub in `gemma4_pooled.py`.
- Gemma 4 12B download stalled/failed twice over the network (xet stall, then an HTTP connection reset); resolved by pointing `--gemma-model` at an already-complete local checkpoint at `/Users/suneel.marti/Desktop/model_cache/gemma-4-12b-it` (verified: `model.safetensors` exactly matches the Hub's reported 23.92 GB).
- Failure-analysis bucketing corrected from a first-word heuristic (only recognized English
  question words; degenerated to a single "other" bucket on the French tabfquad queries) to
  structural features (`has_digit`, `has_at`, `has_color`, `has_quoted_or_acronym`,
  `has_identifier_noun` → `specific_entity_lookup`/`descriptive_general`), and re-run per-dataset
  with each dataset's actual best MUVERA config from `muvera_sweep.csv` instead of one config
  assumed across all four.
