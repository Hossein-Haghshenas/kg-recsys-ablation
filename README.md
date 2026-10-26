# kg-recsys-ablation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23660234.svg)](https://doi.org/10.5281/zenodo.23660234)

Code, configurations and run results for **"When Does the Knowledge Graph Pay? A Controlled Re-Execution Study of Knowledge-Aware Recommendation"**.

The study re-executes five knowledge-aware recommenders (KGAT, KGIN, KGCL, LightKG, CKE) and a knowledge-free reference (LightGCN) under one frozen protocol, and trains every model on several versions of the same knowledge graph:

| condition | knowledge graph given to the model |
|---|---|
| `full` | the original graph |
| `shuf` | a degree-preserving random graph: heads and tails of the triples are permuted independently, so triple count, entity set, per-entity degrees and per-relation counts are kept while the head–tail pairing is destroyed |
| `none` | an empty graph; each model falls back to its own knowledge-free variant (definitions in [`reduce.md`](reduce.md)) |
| `dose25` / `dose50` / `dose75` | a random 25 / 50 / 75 % of the triples (KGAT, KGIN and LightKG only) |

Comparing `full` with `none` measures whether the graph helps at all. Comparing `full` with `shuf` measures whether the *content* of the graph matters, with the number of parameters held fixed.

## Models

| model | implementation used | original code |
|---|---|---|
| LightGCN | RecBole port shipped with LightKG | [gusye1234/LightGCN-PyTorch](https://github.com/gusye1234/LightGCN-PyTorch) |
| KGAT | RecBole port shipped with LightKG | [xiangwang1223/knowledge_graph_attention_network](https://github.com/xiangwang1223/knowledge_graph_attention_network) |
| KGIN | RecBole port shipped with LightKG | [huangtinglin/Knowledge_Graph_based_Intent_Network](https://github.com/huangtinglin/Knowledge_Graph_based_Intent_Network) |
| CKE | RecBole port shipped with LightKG (structural channel only) | — |
| LightKG | authors' model class + RecBole `KGTrainer` | [1371149/LightKG](https://github.com/1371149/LightKG) |
| KGCL | authors' code | [yuh-yang/KGCL-SIGIR22](https://github.com/yuh-yang/KGCL-SIGIR22) |

The LightKG and KGCL source trees are included in `10_models/*/code/official/`, with the changes needed to run them under this protocol. Every change is listed in the model's `PATCHES.md`.

## Repository layout

```
protocol.yaml        frozen experimental protocol (split, metrics, seeds, search budget)
reduce.md            what NONE means for each model
environment.yml      Python environment
00_data/             scripts that prepare the benchmark files and build the KG variants
10_models/
  _shared/           run_recbole.py: trains one RecBole-backed model and writes metrics.json
  <model>/configs/   tuned configuration per dataset and condition
  <model>/runs/      one folder per run: metrics.json, resolved_config.json, training log
  lightkg/code/      LightKG source + RecBole ports of LightGCN, KGAT, KGIN, CKE
  kgcl/              KGCL source, data adapter and runner
20_results/
  runs.csv           one row per run; the only input to the analysis
  hp_search/         every hyper-parameter trial (trial configs, trials.jsonl, best.json)
  hp_space.yaml      search space per model
30_analysis/         paper_statistics.py, make_all.py, stats.py, and generated tables/figures
40_literature_review/ OpenAlex queries, screening registers, Table 10 extraction, and figure scripts
scripts/             hyper-parameter search and batch runner
```

## Setup

Tested on Linux (WSL 2) with one NVIDIA GeForce RTX 5060 Ti, CUDA driver 13.1, Python 3.10.

```bash
conda env create -f environment.yml
conda activate kg-recbole
python scripts/check_env.py
```

If `pip` installs a CPU-only PyTorch, reinstall it for your CUDA version from <https://pytorch.org/get-started/locally/>. KGCL runs in the same environment.

## Data

All models use the benchmark files released with LightKG, so every model sees identical train / validation / test splits (random 80/10/10 per user).

**1. Download** the `dataset/` folder of the LightKG repository into `00_data/raw/`:

```bash
git clone --depth 1 https://github.com/1371149/LightKG.git /tmp/LightKG
mkdir -p 00_data/raw
cp -r /tmp/LightKG/dataset/* 00_data/raw/
```

`00_data/raw/` should now contain `lastfm/`, `book-crossing/`, `ml-1m/` and `Amazon-book/`, each with `.kg`, `.link`, `.train.inter`, `.valid.inter` and `.test.inter` files.

**2. Prepare and check** the files, then build the KG variants:

```bash
python 00_data/build_processed.py     # copies raw files to 00_data/processed/
python 00_data/verify_stats.py        # checks user/item/entity/triple counts against LightKG Table 5
python 00_data/build_variants.py      # writes none, shuf_s0-s2 and dose variants to 00_data/kg_variants/
```

Do not train if `verify_stats.py` reports a mismatch. KGCL converts these files to its own format automatically (`10_models/kgcl/adapt_data.py`).

| dataset | users | items | interactions | entities | relations | triples |
|---|---:|---:|---:|---:|---:|---:|
| Last.FM | 1,873 | 3,847 | 21,173 | 9,367 | 62 | 15,518 |
| Book-Crossing | 11,018 | 9,059 | 24,644 | 77,904 | 27 | 151,500 |
| MovieLens-1M | 6,039 | 3,499 | 573,637 | 77,799 | 51 | 378,151 |

Counts are as RecBole reports them: entity and relation counts include RecBole's padding token, and relations also include the user–item relation.

## Running

All commands run from the repository root.

**One run** (`--condition` is `full`, `none`, `shuf`, `dose25`, `dose50` or `dose75`):

```bash
python 10_models/_shared/run_recbole.py --model lightkg --dataset lastfm --condition full --seed 0
python 10_models/kgcl/run_kgcl.py --dataset bookcrossing --condition shuf --seed 3
```

Each run reads `10_models/<model>/configs/<dataset>__<condition>.yaml` and stops with an error if that file is missing. Results go to `10_models/<model>/runs/<model>__<dataset>__<condition>__s<seed>/metrics.json`.

**Hyper-parameter search.** 15 TPE trials per model × dataset × condition, objective validation Recall@20. The number of negatives per positive (`n_neg` ∈ {1, 2, 5, 10}) is one of the searched parameters. `full` and `none` are searched separately; `shuf` and the dose conditions reuse the `full` winner. The winning configuration is written to `configs/`.

```bash
python scripts/hp_search.py --model kgin --dataset bookcrossing --condition full
```

The search for LightKG / Last.FM / `full` also runs a reproduction check: test Recall@10 of the best trial must fall between 0.26 and 0.33, around the 0.2929 reported in LightKG Table 6. The runs in this repository give 0.2946 to 0.3034 over five seeds.

**All seed runs.** Runs every missing model × dataset × condition × seed job whose configuration exists (5 seeds for the main conditions, 3 for the dose conditions), in batches, checking each result before continuing:

```bash
python scripts/run_core_batch.py --queue remaining --limit 5 --until-empty
```

**Collect and analyse:**

```bash
python 20_results/collect.py            # rebuilds 20_results/runs.csv from every metrics.json
python 30_analysis/paper_statistics.py  # reproduces Tables 16-20 and all statistical tests in the paper
python 30_analysis/make_all.py          # writes exploratory tables and figures to 30_analysis/
```

## Results

`20_results/runs.csv` holds all 321 finished runs across Last.FM, Book-Crossing, and MovieLens-1M (107 runs per dataset: 80 core runs + 27 dose runs), one row per run: test Recall@10/20, NDCG@10/20, MRR@10, best and total epochs, training time, time per epoch, peak GPU memory, parameter count, best validation score and the configuration hash. Evaluation is full ranking over all items.

`30_analysis/paper_statistics.py` reproduces the manuscript's primary measurement tables:

| file | content |
|---|---|
| `paper_cost.csv` | Table 16: capacity, memory and training cost |
| `paper_cells.csv` | Table 17: cell means, standard deviations and CV flags |
| `paper_contrasts.csv` | Table 18: all 45 contrasts (Welch test, 95% CI, Hedges's $g$, Holm correction over 45) |
| `paper_contrasts_pooled_sensitivity.csv` | Table 18 footnote: sensitivity check pooling the stalled LightKG run |
| `paper_dose.csv` | Table 19: dose-response trends across 9 series (exact Jonckheere–Terpstra $p$, Spearman $\rho$) |
| `paper_nneg.csv` | Table 20: negative-sampling spread and winning $n_{\text{neg}}$ per cell |
| `paper_rank.csv` | Section 8.1: exact Mann–Whitney $p$ and complete separation checks |
| `paper_shuf_draws.csv` | Section 8.8: SHUF contrasts re-tested with graph permutation draw as the unit |

In addition, `30_analysis/make_all.py` builds these exploratory summaries:

| file | content |
|---|---|
| `e1_accuracy` | mean ± std of every metric per model × dataset × condition |
| `e2_kg_effect` | paired `full` − `none` differences per seed |
| `e3_full_shuf_none` | `full` / `shuf` / `none` side by side |
| `e4_dose` | accuracy against the fraction of triples kept (Last.FM, Book-Crossing, MovieLens-1M) |
| `e5_cost` | training cost under the equal 15-trial search budget |
| `e6_seed_noise` | seed-to-seed variation per cell |

Notes on the recorded runs:

- **LightKG / Book-Crossing / `shuf`, seed 2** stalled on its initial loss plateau (loss 47.49 at epoch 25, best validation epoch 4, patience 20) and terminated via early stopping without error. In the manuscript's primary analysis (Tables 17 and 18, Section 8.1), this stalled seed is excluded, yielding $n=4$ trained runs with Recall@20 = $0.0861 \pm 0.0021$. When all five seeds are pooled, Recall@20 = $0.0713 \pm 0.0332$ ($n=5$). Both versions are documented in `30_analysis/tables/e3_full_shuf_none.note.txt` and evaluated in `paper_contrasts_pooled_sensitivity.csv`.
- **MovieLens-1M** configurations, 165 search trials (across 11 search cells), and all 107 seed runs are fully executed and included in `runs.csv`.
- The `commit` column records the revision of the LightKG code used for a run. `8c8cc70` is the revision included in this repository.
- File paths inside the logs and JSON records are relative to the repository root.

## Citation

```bibtex
@article{haghshenas2026kgpay,
  title   = {When Does the Knowledge Graph Pay? A Controlled Re-Execution Study of Knowledge-Aware Recommendation},
  author  = {Haghshenas, Hossein and Zare Mehrjardi, Fatemeh and Ghaneai, Hossein and Sardari Zarchi, Mohsen},
  journal = {Journal of Computing and Security},
  year    = {2026},
  note    = {Repository: \url{https://github.com/Hossein-Haghshenas/kg-recsys-ablation}}
}

@software{haghshenas2026kgpay_artifact,
  author    = {Haghshenas, Hossein and Zare Mehrjardi, Fatemeh and Ghaneai, Hossein and Sardari Zarchi, Mohsen},
  title     = {{Hossein-Haghshenas/kg-recsys-ablation: Complete Reproducibility Package --- When Does the Knowledge Graph Pay?}},
  month     = oct,
  year      = {2026},
  publisher = {Zenodo},
  version   = {v1.0.0},
  doi       = {10.5281/zenodo.23660234},
  url       = {https://doi.org/10.5281/zenodo.23660234}
}
```

Please also cite the papers of the models you use, and LightKG for the benchmark files.

## License

The code and result files written for this study are released under the [MIT License](LICENSE).

Third-party code keeps its own terms:

- `10_models/kgcl/code/official/` is © 2022 Yuhao, MIT License ([LICENSE](10_models/kgcl/code/official/LICENSE)).
- `10_models/lightkg/code/official/` comes from [1371149/LightKG](https://github.com/1371149/LightKG), which does not state a license. It is included here, with the changes listed in [`PATCHES.md`](10_models/lightkg/PATCHES.md), for research reproducibility only; all rights remain with its authors.
- The datasets are not redistributed here; they are downloaded from the LightKG repository.
