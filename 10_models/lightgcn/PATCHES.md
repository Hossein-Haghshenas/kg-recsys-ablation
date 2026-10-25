# LightGCN patches

No patches to official LightGCN. Training goes through the RecBole port shipped by LightKG.

- shared runner writes `metrics.json` under `runs/`
- LightGCN ignores KG variants
- LightKG `model/LightGCN.py`: replaced `dok_matrix._update` (removed in scipy 1.13+) with an item assignment loop
