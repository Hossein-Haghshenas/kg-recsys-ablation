# KGAT

- **Implementation:** RecBole port shipped with LightKG, `code/official/model/KGAT.py` in [`../lightkg`](../lightkg)
- **Original code:** https://github.com/xiangwang1223/knowledge_graph_attention_network (TensorFlow 1.12, not executed)
- **NONE:** `G = G_ui`, relation set = {interact}, no `L_kg` term (see [`reduce.md`](../../reduce.md))
- **Changes:** [PATCHES.md](PATCHES.md)

```bash
python 10_models/_shared/run_recbole.py --model kgat --dataset lastfm --condition full --seed 0
```
