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

Bootstrap 95% CIs (`scripts/bootstrap_ci.py`, 2000 resamples, seed=0) on the composite bucket gap
and each individual structural feature, resampling each group independently within dataset:

| dataset | split | n (group A / B) | diff | CI | excludes 0 |
|---|---|---|---|---|---|
| docvqa | bucket: specific_entity_lookup vs descriptive_general | 55/145 | +0.10 | [-0.73, +1.00] | no |
| docvqa | has_digit (present vs absent) | 26/174 | -0.03 | [-1.09, +1.22] | no |
| docvqa | has_at (present vs absent) | 0/200 | - | too few samples | - |
| docvqa | has_color (present vs absent) | 2/198 | - | too few samples | - |
| docvqa | has_quoted_or_acronym (present vs absent) | 27/173 | +0.05 | [-0.99, +1.29] | no |
| docvqa | has_identifier_noun (present vs absent) | 4/196 | +0.29 | [-1.30, +2.59] | no |
| arxivqa | bucket: specific_entity_lookup vs descriptive_general | 96/104 | +0.31 | [-0.31, +0.95] | no |
| arxivqa | has_digit (present vs absent) | 48/152 | -0.49 | [-0.99, -0.01] | yes |
| arxivqa | has_at (present vs absent) | 0/200 | - | too few samples | - |
| arxivqa | has_color (present vs absent) | 17/183 | +2.54 | [+0.80, +4.54] | yes |
| arxivqa | has_quoted_or_acronym (present vs absent) | 43/157 | -0.40 | [-0.95, +0.21] | no |
| arxivqa | has_identifier_noun (present vs absent) | 0/200 | - | too few samples | - |
| infovqa | bucket: specific_entity_lookup vs descriptive_general | 81/119 | +0.04 | [-0.56, +0.65] | no |
| infovqa | has_digit (present vs absent) | 49/151 | -0.52 | [-0.95, -0.10] | yes |
| infovqa | has_at (present vs absent) | 0/200 | - | too few samples | - |
| infovqa | has_color (present vs absent) | 4/196 | +0.71 | [-2.41, +5.38] | no |
| infovqa | has_quoted_or_acronym (present vs absent) | 31/169 | +0.03 | [-0.87, +1.08] | no |
| infovqa | has_identifier_noun (present vs absent) | 5/195 | +3.33 | [-0.28, +7.05] | no |
| tabfquad | bucket: specific_entity_lookup vs descriptive_general | 126/74 | +0.63 | [-0.09, +1.37] | no |
| tabfquad | has_digit (present vs absent) | 104/96 | +0.50 | [-0.28, +1.31] | no |
| tabfquad | has_at (present vs absent) | 0/200 | - | too few samples | - |
| tabfquad | has_color (present vs absent) | 0/200 | - | too few samples | - |
| tabfquad | has_quoted_or_acronym (present vs absent) | 47/153 | -0.03 | [-0.89, +0.88] | no |
| tabfquad | has_identifier_noun (present vs absent) | 2/198 | - | too few samples | - |

**The composite `specific_entity_lookup` vs `descriptive_general` gap does not exclude zero in any
of the four datasets** — the consistent positive point-estimate direction reported in an earlier
version of this table is not statistically distinguishable from noise at n=200/dataset. Of the 20
individual-feature comparisons with enough samples to test, exactly three exclude zero:
**arxivqa `has_digit`** (diff −0.49, i.e. digit-containing queries do *better* under MUVERA there,
the opposite of the hypothesized direction), **arxivqa `has_color`** (diff +2.54, confirms the
hypothesized direction), and **infovqa `has_digit`** (diff −0.52, also opposite direction). Every
other individual feature does not exclude zero, or has too few samples to test (`has_at` has zero
matches in all four datasets; `has_color`/`has_identifier_noun` have zero or near-zero matches in
tabfquad).

### Pooled across datasets (stratified bootstrap, `scripts/bootstrap_pooled.py`)

The per-dataset tests above are underpowered for the rarer features (`has_color` n=17/2/4/0 across
the four datasets, `has_identifier_noun` n=4/0/5/2). This pools each feature's group A/B across all
four datasets — resampling each dataset's own groups independently first, then combining — so an
underpowered feature gets one properly-powered test instead of four weak ones. A dataset only
contributes to a feature's pool if it has ≥5 rows in that group (`MIN_PER_GROUP=5`); the
per-dataset column shows exactly which datasets qualified.

| feature | per-dataset n (present) | pooled n (present/absent) | diff | CI | excludes 0 |
|---|---|---|---|---|---|
| has_digit | docvqa=26, arxivqa=48, infovqa=49, tabfquad=104 | 227/573 | -0.20 | [-0.57, +0.18] | no |
| has_at | none | insufficient data across all datasets | - | - | - |
| has_color | docvqa=2(excluded,<5), arxivqa=17, infovqa=4(excluded,<5) | 17/183 | +2.54 | [+0.82, +4.46] | yes |
| has_quoted_or_acronym | docvqa=27, arxivqa=43, infovqa=31, tabfquad=47 | 148/652 | -0.16 | [-0.57, +0.29] | no |
| has_identifier_noun | docvqa=4(excluded,<5), infovqa=5, tabfquad=2(excluded,<5) | 5/195 | +3.33 | [-0.24, +7.17] | no |
| bucket: specific_entity_lookup vs descriptive_general | docvqa=55, arxivqa=96, infovqa=81, tabfquad=126 | 358/442 | +0.15 | [-0.20, +0.52] | no |

**After pooling, the only effect that excludes zero is `has_color` — and it is driven entirely by
arxivqa.** Only arxivqa had enough color-referencing queries (n=17) to clear the ≥5-per-group
threshold; docvqa (n=2) and infovqa (n=4) were excluded from the pool as too few, and tabfquad had
zero such queries. So "has_color is harder for MUVERA" is not a cross-dataset finding — it is an
arxivqa finding, full stop, and should be reported as such rather than "confirmed across datasets."
`has_digit` is different: arxivqa (−0.49) and infovqa (−0.52) individually exclude zero in the
*same* direction — digit-containing queries are easier for MUVERA in both, a real, replicated
effect on its own terms. But this does not generalize across all four domains: tabfquad's point
estimate points the opposite way (+0.50, not itself significant) and docvqa sits near zero (−0.03),
and pooling all four in washes out the arxivqa/infovqa signal rather than confirming it. This is a
domain-conditional effect, not a universal MUVERA property — the arxivqa/infovqa result isn't
noise, it just doesn't hold everywhere. The composite `specific_entity_lookup` vs `descriptive_general`
bucket still does not exclude zero even with full pooled power (358 vs 442 rows). **The honest
mechanism-level finding from this run is: `has_color` is harder for MUVERA in arxivqa; nothing else
in the structural-feature set survives pooling.** That is a narrower claim than "specific-entity
lookups are systematically harder than descriptive questions," and the data doesn't support the
broader one.

Full per-dataset console output below (pre-bootstrap, i.e. point estimates only).

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
