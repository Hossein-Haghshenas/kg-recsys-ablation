# `NONE` definitions

These definitions were fixed before any training run and were not changed afterwards.

`NONE` means `G_kg = ∅`. The collaborative graph `G_ui` stays intact. Do not replace a model's `NONE` variant with LightGCN.

| model | `NONE` means | reduces to |
|---|---|---|
| LightGCN | no KG component | itself (reference) |
| CKE | `e_i = e_i^CF`, drop `L_kg` | BPR-MF |
| KGAT | `G = G_ui`, relation set = {interact}, drop `L_kg` | NGCF with attention |
| KGIN | drop relations; intents become free vectors | disentangled propagation on `G_ui` |
| KGCL | knowledge-guided augmentation → random dropout; drop knowledge view | approximately SGL |
| KGRec | drop triple scoring and masking | LightGCN + InfoNCE |
| LightKG | relation scalars = 1; no knowledge contrastive term | LightGCN + contrastive layer |

## implementation notes

- **empty KG file:** RecBole `KnowledgeBasedDataset` needs a valid `.kg` header. `NONE` writes a header-only file. if a model still requires one triple, add a documented dummy self-loop and record it in `PATCHES.md`.
- **SHUF:** same node and edge counts; head and tail of each triple are independently permuted. build three files: `shuf_s0`, `shuf_s1`, `shuf_s2`.
- **DOSE:** keep 25 / 50 / 75 percent of triples at random, then force-keep one original triple per `.link` entity so RecBole 1.1.1 does not drop item–entity maps (and their interactions). only KGAT, KGIN, LightKG. three seeds each.
- **LightGCN:** ignore KG variants. one condition (`full`) is enough; `none`/`shuf` are not scheduled.

