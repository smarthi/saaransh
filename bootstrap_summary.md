# Bootstrap 95% CIs on rank_delta gaps (2000 resamples, seed=0)

Positive diff = specific/feature-present group lost more ground to ceiling than the comparison group. "excludes 0" = the interval doesn't cross zero at this CI, i.e. the gap survives resampling noise; it does NOT by itself mean the effect is large or practically important, only that it's not indistinguishable from zero at this sample size.

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
