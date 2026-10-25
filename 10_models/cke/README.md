# CKE

Structural-only CKE (no textual or visual channels).

- **Implementation:** RecBole port shipped with LightKG, `code/official/model/CKE.py` in [`../lightkg`](../lightkg)
- **NONE:** empty knowledge graph, so the model reduces to BPR-MF (see [`reduce.md`](../../reduce.md))
- **Changes:** [PATCHES.md](PATCHES.md)

```bash
python 10_models/_shared/run_recbole.py --model cke --dataset lastfm --condition full --seed 0
```
