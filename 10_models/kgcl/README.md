# KGCL

- **Implementation:** the authors' SIGIR 2022 code, in [`code/official`](code/official) (MIT licence, [LICENSE](code/official/LICENSE))
- **Original code:** https://github.com/yuh-yang/KGCL-SIGIR22
- **Data:** `adapt_data.py` converts the RecBole benchmark files into KGCL's `train.txt` / `valid.txt` / `test.txt` / `kg.txt` format
- **NONE:** knowledge-guided augmentation replaced by random dropout and the knowledge view removed, which is approximately SGL (see [`reduce.md`](../../reduce.md))
- **Changes:** [PATCHES.md](PATCHES.md)

```bash
python 10_models/kgcl/run_kgcl.py --dataset lastfm --condition full --seed 0
```
