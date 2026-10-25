# KGAT patches

Official TensorFlow 1.12 code is not executed. All runs use LightKG's RecBole `model/KGAT.py`.

- `NONE` is implemented by swapping in a header-only KG file, not by editing the model class
- if RecBole rejects an empty KG, a dummy self-loop on an item-linked entity is added and recorded here
- dummy none triple must use a `.link` entity; a raw KG head (e.g. Book-Crossing `14967`) is not in `entity2item` and crashes remapping
- dose graphs must retain every `.link` entity; a raw 25/50/75% sample hits the same remapping KeyError and would also drop those items' interactions
- LightKG `model/KGAT.py`: DGL graph build replaced with scipy COO (DGL 2.1 is broken on torch 2.11 / torchdata 0.11)
