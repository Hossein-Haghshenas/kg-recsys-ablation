# LightKG patches

- shared runner instantiates `LightKG` from the official repo instead of `main.py`
- `LightKG.py`: local `spmm` so the model does not need `torch_sparse`
- `LightKG.py`: KG degree tensor uses `n_entities` (dummy NONE triple does not cover every id)
- `LightKG.py`: store uncoalesced relation ids; coalesced CKG values are ones so duplicate `(h,t)` pairs cannot sum ids past the embedding table (CUDA crash on Last.FM `full`)
- `LightKG.py`: similarity is the ui subgraph with an explicit size; unsized `sparse.mm` densified `n_users+n_entities` and OOM'd Book-Crossing (~30gb)
- `LightKG.py`: `spmm` uses `index_add` so GNN backward does not allocate a dense CKG (same 30gb OOM on the first backward)
- `NONE` feeds one dummy self-loop on an item-linked entity so RecBole can remap `entity2item`
- seed handling follows `protocol.yaml`, not the hardcoded 2020/200 split in upstream `main.py`
- `LightKG.py`: the test-time embedding placeholders set in `__init__` are detached copies, so the model also runs on CPU (on CUDA they were already plain tensors; they are overwritten before use)
