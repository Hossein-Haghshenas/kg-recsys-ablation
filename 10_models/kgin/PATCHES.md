# KGIN patches

Runs use LightKG's RecBole `model/KGIN.py`. Official WWW'21 code is a reference submodule only.

- `NONE` uses the empty-KG variant from `00_data/kg_variants`
- intent count and other hyperparameters stay at the FULL config until a dedicated NONE search runs
- LightKG `model/KGIN.py`: replaced `dok_matrix._update` with an item assignment loop for scipy 1.15
- LightKG `model/KGIN.py`: local `scatter_mean` so KGIN does not need `torch_scatter`
- LightKG `model/KGIN.py`: dropped `self.loss += mf_loss` (it kept the GCN graph across batches)
