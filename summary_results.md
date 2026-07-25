# saaransh gap-filling benchmark — results

ColQwen2 MaxSim (exact ceiling) vs MUVERA (single-vector FDE) vs Gemma 4 12B pooled, on 4 ViDoRe datasets (docvqa, arxivqa, infovqa, tabfquad; 200 docs / 200 queries each).

## MUVERA vs exact-MaxSim ceiling

| dataset | ceiling nDCG@5 | best MUVERA nDCG@5 | % of ceiling | best config | storage/page | ceiling-lost-by-MUVERA |
|---|---|---|---|---|---|---|
| arxivqa | 0.895 | 0.830 | 93% | muvera[DE\|k8\|r8\|c-] | 1024 KB | 19/200 lost |
| docvqa | 0.655 | 0.534 | 82% | muvera[CA\|k8\|r8\|c-] | 1024 KB | 29/200 lost |
| infovqa | 0.914 | 0.839 | 92% | muvera[CA\|k8\|r8\|c-] | 1024 KB | 23/200 lost |
| tabfquad | 0.594 | 0.557 | 94% | muvera[DE\|k8\|r4\|c-] | 512 KB | 29/200 lost |

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

## Failure analysis — per dataset (MUVERA config: calibrated/default_identity k8, r8/r4, best from sweep)

### docvqa

```
ceiling got it right but MUVERA lost it: 29/200 queries

avg rank_delta by first-word bucket (higher = MUVERA relatively worse):
  where    n=2    avg_delta=+2.00
  other    n=9    avg_delta=+1.44
  when     n=9    avg_delta=+1.44
  what     n=135  avg_delta=+1.29
  which    n=20   avg_delta=+0.95
  who      n=20   avg_delta=+0.95
  how      n=5    avg_delta=+0.20

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

### arxivqa

```
ceiling got it right but MUVERA lost it: 19/200 queries

avg rank_delta by first-word bucket (higher = MUVERA relatively worse):
  what     n=91   avg_delta=+0.90
  other    n=87   avg_delta=+0.76
  which    n=20   avg_delta=+0.05
  how      n=2    avg_delta=+0.00

corr(query length, rank_delta) = -0.113

worst 20 regressions (ceiling rank vs MUVERA rank):
  delta= 10  ceiling=   1  muvera=None  "Based on the vector fields shown in the bottom portion of the figure, what characteristic changes from panel A to panel D?"
  delta= 10  ceiling=   1  muvera=None  "Based on the phase space plots in figure (b), what can be inferred about the state of the system at point 2?"
  delta= 10  ceiling=   1  muvera=None  "If the colors in the image represent different categories, what can be inferred about the category represented by the color cyan?"
  delta= 10  ceiling=   1  muvera=None  "What is the role of the coordinate \( p \) in the figure?"
  delta=  9  ceiling=   2  muvera=None  "What does the dashed red line in the figure represent?"
  delta=  9  ceiling=   2  muvera=None  "What does the solid red line represent in the figure?"
  delta=  9  ceiling=   2  muvera=None  "Based on the geometry shown in figure (a), what does the blue rhombus most likely represent?"
  delta=  9  ceiling=   1  muvera=  10  "What does the graph suggest about the scaling of different operators in the large-N limit?"
  delta=  9  ceiling=   2  muvera=None  "What does panel (a) of the figure primarily illustrate?"
  delta=  8  ceiling=   3  muvera=None  "At what approximate x-axis value does the purple diamond dataset begin to level off?"
  delta=  6  ceiling=   1  muvera=   7  "What does the contour plot within the image most likely represent?"
  delta=  5  ceiling=   6  muvera=None  "What does the dashed line represent in this layered network diagram?"
  delta=  5  ceiling=   6  muvera=None  "What is the significance of the two different sizes mentioned in figure (c)?"
  delta=  5  ceiling=   3  muvera=   8  "Based on the figure, what can be inferred about the location of the observed subject?"
  delta=  5  ceiling=   1  muvera=   6  "What can be inferred about the stability of the coefficients over time when comparing the plots in figures A and E?"
  delta=  5  ceiling=   1  muvera=   6  "What does the red dashed line with square markers in the graph most likely represent?"
  delta=  5  ceiling=   1  muvera=   6  "If you were to extrapolate the trend observed in the graph, which of the following outcomes would be most likely at temperatures slightly above 3 K?"
  delta=  4  ceiling=   1  muvera=   5  "Based on the behavior of the functions in the figure, what is a likely physical context of the graphs?"
  delta=  4  ceiling=   7  muvera=None  "What does the color scale in panel (b) represent?"
  delta=  4  ceiling=   2  muvera=   6  "Based on the figure, which statement is true regarding the direction of wave propagation?"
```

### infovqa

```
ceiling got it right but MUVERA lost it: 23/200 queries

avg rank_delta by first-word bucket (higher = MUVERA relatively worse):
  who      n=8    avg_delta=+1.75
  which    n=45   avg_delta=+0.84
  what     n=90   avg_delta=+0.63
  other    n=20   avg_delta=+0.35
  when     n=5    avg_delta=+0.00
  how      n=32   avg_delta=-0.16

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

### tabfquad

```
ceiling got it right but MUVERA lost it: 29/200 queries

avg rank_delta by first-word bucket (higher = MUVERA relatively worse):
  other    n=200  avg_delta=+0.81

corr(query length, rank_delta) = -0.139

worst 20 regressions (ceiling rank vs MUVERA rank):
  delta= 10  ceiling=   1  muvera=None  "De combien étaient les ventes en millions d'euros en 2011 ?"
  delta= 10  ceiling=   1  muvera=None  "quel est le total de passifs au 31 decembre 2011 ?"
  delta= 10  ceiling=   1  muvera=None  "Combien de zones distingue t on ?"
  delta= 10  ceiling=   1  muvera=None  "Quelles sont les dépenses en acquisition de terrain en 2008 ?"
  delta=  9  ceiling=   2  muvera=None  "Quelles sont les différentes méthodes de calcul des primes dans les entreprises françaises ?"
  delta=  9  ceiling=   2  muvera=None  "Quel était le montant total des litiges en euros enregistré en juin 2017?"
  delta=  9  ceiling=   2  muvera=None  "Combien d'actions ont été vendues entre décembre 2010 et mars 2011 ?"
  delta=  9  ceiling=   2  muvera=None  "Quelles étaient les principales catégories de dette financière d'une entreprise à la fin de l'année 2019 et 2020?"
  delta=  7  ceiling=   4  muvera=None  "Quelles étaient les valeurs des actifs financiers disponibles à la vente au 31 décembre 2011 ?"
  delta=  7  ceiling=   4  muvera=None  "Y a-t-il des années où certaines entreprises de transport telles que ASF ou COFIROUTE n'ont pas enregistré de croissance ?"
  delta=  7  ceiling=   4  muvera=None  "Comment l'utilisation des différents types de rappels a-t-elle évolué d'une année à l'autre?"
  delta=  7  ceiling=   3  muvera=  10  "Quel était le résultat avant impots des caisses régionales en 2016 ?"
  delta=  6  ceiling=   1  muvera=   7  "Peut-on trouver des exemples de structures de primes basées sur l'égalité ou la performance dans les grandes entreprises ?"
  delta=  5  ceiling=   4  muvera=   9  "Quels sont les champs typiques et leur signification dans une base de données d'exercices financiers d'une collectivité ?"
  delta=  5  ceiling=   2  muvera=   7  "Comment varient les plafonds de ressources pour l'accès aux logements sociaux entre Paris, l'Île-de-France et les autres régions françaises ?"
  delta=  5  ceiling=   5  muvera=  10  "Quelles sont les proportions de populations dans les villes selon différentes tranches de taille d'habitants?"
  delta=  4  ceiling=   7  muvera=None  "Comment a évolué le résultat net des sociétés consolidées dans le secteur aérien entre 2011 et 2012?"
  delta=  4  ceiling=   1  muvera=   5  "Quel est le montant des dotations aux amortissements et aux provisions ? "
  delta=  4  ceiling=   4  muvera=   8  "Quelles étaient les fourchettes de pourcentage pour les années 2006 à 2008 concernant un indicateur économique par zone géographique?"
  delta=  4  ceiling=   4  muvera=   8  "Sur l'année 2009-2010, de combien de % ont augmenté les droits de vol ?"
```

## Run notes (fixes applied to get here, not results)

- `scripts/failure_analysis.py` was misplaced at `src/saaransh/failure_analysis.py`; moved to match `run.sh` and its own docstring.
- `run.sh`'s `COLQWEN_MODEL`/`GEMMA_MODEL` pointed at nonexistent local paths; fixed to use real HF repo ids + `--cache-dir`.
- Full pytest suite (20 passed, 1 skipped — unrelated missing `pdf` extra) required `KMP_DUPLICATE_LIB_OK=TRUE` to avoid a faiss+torch duplicate-OpenMP crash on macOS.
- Gemma precision sweep restricted to `bf16`; MLX q8/q4 hidden-state extraction is an unimplemented stub in `gemma4_pooled.py`.
- Gemma 4 12B download stalled/failed twice over the network (xet stall, then an HTTP connection reset); resolved by pointing `--gemma-model` at an already-complete local checkpoint at `/Users/suneel.marti/Desktop/model_cache/gemma-4-12b-it` (verified: `model.safetensors` exactly matches the Hub's reported 23.92 GB).
