# Literature Review Materials and Audit Trail

This directory contains the data, registers, extraction sheets, and figure-generation scripts supporting the literature analysis and verification search of **"When Does the Knowledge Graph Pay? A Controlled Re-Execution Study of Knowledge-Aware Recommendation"** (Section 2, Section 7, and Declarations).

## Contents

- `openalex_queries.json`: Verbatim REST API query URLs, parameters, and hit counts for OpenAlex queries O1 through O4 executed on 26 August 2026.
- `stage1b_screening_register.csv`: Complete screening register of all 4,669 records entering Stage 1b, with publication metadata and screening decisions (`retrieve` vs. `exclude` with exclusion codes F1-F4).
- `retrieval_candidates_1912.csv`: The list of 1,912 candidate reports sought for retrieval after passing Stage 1b relevance screening.
- `table10_transparency_extraction.csv`: The extraction sheet behind Table 10 (Reporting transparency of 21 representative models across Paradigms I to V), documenting train/test split disclosure, evaluation regime reporting, code availability, and knowledge-graph ablation status.
- `prisma_counts.json`: Audit trail records reconciling every identification, deduplication, and screening count in the PRISMA 2020 flow diagram (Figure 3 / `fig0_prisma.pdf`).
- `scripts/`:
  - `rebuild_fig0_prisma.py`: Generates Figure 3 (`fig0_prisma.pdf`).
  - `rebuild_fig2_pipeline.py`: Generates Figure 1 (`fig2_pipeline.pdf`), the unified architecture template across Paradigms I–V.

## OpenAlex Queries (O1–O4)

| ID | Date Range | Query String | Returned Records |
|---|---|---|---:|
| O1 | 2013-01-01 to 2026-08-26 | `"knowledge graph" AND recommendation` | 5,520 |
| O2 | 2013-01-01 to 2026-08-26 | `"knowledge-aware" AND recommendation` | 788 |
| O3 | 2013-01-01 to 2026-08-26 | `"knowledge graph" AND recommendation AND (ablation OR reproducibility OR benchmark OR evaluation)` | 1,950 |
| O4 | 2021-01-01 to 2026-08-26 | `"knowledge graph" AND recommendation AND (contrastive OR self-supervised OR diffusion OR transformer OR "large language model")` | 1,203 |

All queries can be re-executed verbatim via the URLs in `openalex_queries.json`.
