# LightKG

- **Implementation:** the LightKG model class trained with RecBole `KGTrainer`, in [`code/official`](code/official)
- **Original code:** https://github.com/1371149/LightKG (includes the benchmark files used by every model here)
- **NONE:** relation scalars fixed to 1, no knowledge contrastive term (see [`reduce.md`](../../reduce.md))
- **Changes:** [PATCHES.md](PATCHES.md)

`code/official/model/` also holds the RecBole ports of LightGCN, KGAT, KGIN and CKE used by the other model folders.

```bash
python 10_models/_shared/run_recbole.py --model lightkg --dataset lastfm --condition full --seed 0
```
