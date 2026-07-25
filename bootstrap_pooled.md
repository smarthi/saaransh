# Pooled cross-dataset bootstrap 95% CIs (stratified, 3000 resamples, seed=0)

Each dataset's own group is resampled independently, then pooled across datasets, before computing the mean diff per bootstrap iteration — a dataset can't dominate the estimate just by having more rows with the feature present. A dataset only contributes to a feature's pool if it has at least 5 rows in that group; per-dataset counts are shown so you can see exactly which datasets are actually driving each pooled result.

| feature | per-dataset n (present) | pooled n (present/absent) | diff | CI | excludes 0 |
|---|---|---|---|---|---|
| has_digit | docvqa=26, arxivqa=48, infovqa=49, tabfquad=104 | 227/573 | -0.20 | [-0.57, +0.18] | no |
| has_at | none | insufficient data across all datasets | - | - | - |
| has_color | docvqa=2(excluded,<5), arxivqa=17, infovqa=4(excluded,<5) | 17/183 | +2.54 | [+0.82, +4.46] | yes |
| has_quoted_or_acronym | docvqa=27, arxivqa=43, infovqa=31, tabfquad=47 | 148/652 | -0.16 | [-0.57, +0.29] | no |
| has_identifier_noun | docvqa=4(excluded,<5), infovqa=5, tabfquad=2(excluded,<5) | 5/195 | +3.33 | [-0.24, +7.17] | no |
| bucket | docvqa=55, arxivqa=96, infovqa=81, tabfquad=126 | 358/442 | +0.15 | [-0.20, +0.52] | no |
