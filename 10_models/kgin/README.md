# KGIN

- **Implementation:** RecBole port shipped with LightKG, `code/official/model/KGIN.py` in [`../lightkg`](../lightkg)
- **Original code:** https://github.com/huangtinglin/Knowledge_Graph_based_Intent_Network (not executed)
- **NONE:** relations dropped; intents become free vectors (see [`reduce.md`](../../reduce.md))
- **Changes:** [PATCHES.md](PATCHES.md)

```bash
python 10_models/_shared/run_recbole.py --model kgin --dataset lastfm --condition full --seed 0
```
