# KGCL patches

- RecBole LightKG files are converted to KGCL `train.txt` / `valid.txt` / `test.txt` / `kg.txt` by `adapt_data.py`
- `adapt_data.py`: map KG nodes through `.link`; flip non-item-headed triples so TransE `embedding_item(h)` stays in range (Book-Crossing authors are heads)
- `NONE` runs official `--model sgl` (knowledge view off, random UI dropout)
- official torch 1.8.1 cannot run on RTX 5060 Ti; the env uses a newer torch
- `world.py` / `register.py`: added `lastfm`, `bookcrossing`, `amazonbook` to the Loader path (not the social LastFM class)
- `main.py`: removed unused `torch_sparse` import
- `main.py`: import `tensorboardX` only when `--tensorboard=1` (the environment does not install it)
- `main.py`: early stop on scalar `recall@20` when `--topks=[10,20]` (numpy cannot truth-test the array)
- `main.py` / `Procedure.py` / `dataloader.py`: early-stop and search objective use **valid** Recall@20 (`valid.txt`), not test. `eval_ckpt.py` still reports test. patience 20 for lastfm/bookcrossing
- `Procedure.py`: report `mrr@k` (official `MRRatK_r` is not reciprocal rank)
- `run_kgcl.py`: parse logs / eval the saved checkpoint into `metrics.json`; `--run-id`; always load `{dataset}__{condition}.yaml`; write `best_valid_score`
- `parse.py` / `world.py` / `main.py` / `Procedure.py`: `--neg_k` so n_neg can enter the equal-budget search; extra negatives are uniform BPR terms
- `run_kgcl.py`: refuse a missing `{dataset}__{condition}.yaml` (no lastfm fallback)
- `Procedure.py`: keep the last TransE batch when unique KG heads < 4096 (Last.FM)
- `contrast.py`: removed `torch_sparse` / `torch_geometric`; local `degree()` replacement
