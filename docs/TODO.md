# TODO

What's still open, as of 2026-10-29. Run commands from the repo root after
`source ../.venv/bin/activate`. Finished items live in [NOTES.md](NOTES.md);
the old step-by-step list (recording, downloads, training) is
[archive/HUMAN_TODO.md](archive/HUMAN_TODO.md).

## Test once, when a model is final

No test split has been touched. Run each command a single time, once nothing
about that model will change again, and record the numbers in NOTES.md. The
word and J/Z scripts refuse a second test run (`results/*_test_used*.json`).
`evaluate.py` doesn't, so for letters this rule is on you.

```bash
# letters
python src/evaluate.py --checkpoint checkpoints/mlp_pugeault_best.pt --split test

# J/Z (test session m5)
python src/diagnose_motion.py --checkpoint checkpoints/motion_v2_best.pt --split test

# words: the 250-sign model on GISLR and ASL Citizen, then the full-vocabulary model
python src/evaluate_words.py --checkpoint checkpoints/words_transformer_full_aslc_r3_best.pt --split test
python src/evaluate_words.py --checkpoint checkpoints/words_transformer_full_aslc_r3_best.pt --split test \
    --extra-cache-dir data/processed/asl_citizen --dataset asl_citizen
python src/evaluate_words.py --checkpoint checkpoints/words_transformer_full_aslc_all_r3_best.pt --split test \
    --extra-cache-dir data/processed/asl_citizen_all --dataset asl_citizen_all
```

## Next levers

- **More data beats more settings.** Tuning has plateaued (NOTES.md, "ASL
  Citizen fixes two-handed signs"). What moves the needle is examples of more
  signers, or more taught examples per SignDesk profile.
- **Segmentation was tuned on the clips it's scored on.** The 0.75 auto-segmented
  number is a bit optimistic. Record a fresh batch of word clips to check it.
- **Two-handed signs live** fail mostly on detection, not classification:
  Holistic misses one of two touching hands. Framing advice is in the README.
- **`data/` vs `src/data/` name clash.** `import data` only works because every
  script puts `src/` first on `sys.path`. Renaming the package (or the dataset
  folder) would remove the bootstrap boilerplate. It's a repo-wide refactor,
  so it hasn't been done.
