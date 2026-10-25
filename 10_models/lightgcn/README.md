# LightGCN

Non-knowledge-graph reference model. Only the `full` condition is run, because LightGCN does not read the knowledge graph.

- **Implementation:** RecBole port shipped with LightKG, `code/official/model/LightGCN.py` in [`../lightkg`](../lightkg)
- **Original code:** https://github.com/gusye1234/LightGCN-PyTorch (not executed)
- **Changes:** [PATCHES.md](PATCHES.md)

```bash
python 10_models/_shared/run_recbole.py --model lightgcn --dataset lastfm --condition full --seed 0
```
