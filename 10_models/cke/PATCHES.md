# CKE patches

- code path is LightKG RecBole `model/CKE.py`, not the 2016 Theano original
- structural KG only; textual/visual CKE channels are not implemented
- `NONE` uses the shared dummy-KG swap in `run_recbole.py`, not a BPR-MF reimplementation
