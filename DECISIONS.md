# Decisions log

Append-only. One entry per judgement call made without Ben available, with the
reasoning in a line or two. Newest at the bottom.

## 2026-09-28 (autonomous session, branch `feature/motion-v2-and-words`)

- **GISLR download blocked (403), no mirror used.** `kaggle competitions files -c asl-signs`
  works (auth fine) but every download call (bulk and single-file, 3 attempts) returns
  403 Forbidden, the signature of unaccepted competition rules. Public Kaggle mirrors
  were checked (`markwijkhuizen/gislr-dataset-public`, `cristaliss/islr-hands-only-dataset`,
  `aapokossi/saved-tfdataset-...`): all are someone else's *preprocessed* tensors
  (dominant-hand only, fixed landmark subset, no participant ids), so they can't answer
  the two-hand ablation or feed a pipeline whose landmark indices must match the live
  demo. Logged in HUMAN_TODO; Phases B–E are built and verified against a synthetic
  fixture with the exact competition schema instead.
- **Read parquet straight out of the zip, don't extract.** The competition archive is
  roughly 40+ GB and extraction would need a similar amount again; with ~96 GB free that
  breaks the ">20 GB free after extraction" rule. `preprocess_gislr.py` reads members
  from the zip (or an extracted directory, whichever exists), so the extracted copy is
  never needed.
