# HUMAN_TODO

Things only Ben can do, in priority order. Run every command from the repo root
after `source ../.venv/bin/activate`. Created 2026-09-28 by the autonomous run on
branch `feature/motion-v2-and-words` (see AUTONOMOUS_RUN_REPORT.md).

---

## 1. Record J / Z / NONE motion v2 data

The v1 J/Z model learned handshape, not motion (NOTES.md, "J/Z motion v1
diagnosis"). v2 needs new recordings because v1 clips threw the trajectory away
at capture time. Old v1 clips in `data/motion/` stay as they are and are not used.

**Plan: 5 sessions** (3 is the minimum, but with 3 the signer split is 1 train /
1 val / 1 test, so the model would train on a single session). Use different days,
lighting, clothing or seating distance for each. Give each session its **own
signer id**, because that's the unit the split holds out:

```bash
python src/data/capture_motion_v2.py --signer ben_m1 --session s1_desk_daylight
python src/data/capture_motion_v2.py --signer ben_m2 --session s2_desk_evening
python src/data/capture_motion_v2.py --signer ben_m3 --session s3_standing
python src/data/capture_motion_v2.py --signer ben_m4 --session s4_far_from_cam
python src/data/capture_motion_v2.py --signer ben_m5 --session s5_other_room
```

Each session defaults to 20 clips each of J, Z and NONE, at 2.0 s per clip
(about 10 minutes). Controls: SPACE records (1.5 s countdown first), r redoes the
last clip, n moves to the next label, q quits (anything already captured is kept).

- **J and Z:** start the clip with the hand still, trace the letter once, then
  stop. Vary speed and size a little between clips. Don't retrace mid-clip.
- **NONE:** follow the on-screen prompt. It cycles through `hold_I` (hold I still,
  no hook), `hold_D` (hold D/index-up still, no zigzag), `hold_static` (hold a named
  letter), `between_letters` (move from one letter to another as when spelling),
  and `wave_reposition`. The two hold prompts matter most: they're the negatives
  v1 would have called J and Z.
- Clips with fewer than 10 frames of detected hand are thrown away on the spot
  and the prompt repeats.

Success: `data/motion_v2/motion_v2_metadata.csv` has about 300 rows across 5
signer ids, and

```bash
python -c "import pandas as pd; d=pd.read_csv('data/motion_v2/motion_v2_metadata.csv'); print(d.groupby(['signer_id','label']).size().unstack()); print(d[d.label=='NONE'].subtype.value_counts())"
```

shows roughly 20/20/20 per signer and every NONE sub-type represented.

## 2. Train and diagnose the v2 motion model

```bash
python src/diagnose_motion.py                       # probes only, no model yet
python src/train_motion_v2.py --epochs 80 2>&1 | tee logs/motion_v2.log
python src/diagnose_motion.py --checkpoint checkpoints/motion_v2_best.pt
```

What "good" looks like in the diagnosis output (validation split):

| check | good | bad (the v1 failure) |
|---|---|---|
| probe `first_frame` / `mean_frame` | clearly below `trajectory_only` (hold_I vs J share a handshape) | ~1.00 |
| probe `trajectory_only` | high (≥ 0.85) | near chance |
| ablation `time_shuffled` | clearly below `normal` | ≈ `normal` |
| ablation `motion_neutralised` | large drop from `normal` | ≈ `normal` |
| ablation `handshape_neutralised` | small drop | large drop |
| false positives `hold_I->J`, `hold_D->Z` | < 10 % each | high |

If that looks right, try it live:

```bash
python src/demo.py --checkpoint checkpoints/mlp_pugeault_best.pt \
    --motion-checkpoint checkpoints/motion_v2_best.pt
```

Terminal lines start with `[motion]`. Signing ordinary letters and moving
between them should give `NONE ... released` (or never trigger), not a J or Z.
If slow J/Z fail to trigger, lower `--motion-threshold` (the default is 1.0
hand-lengths/second; the `triggered, speed …` prints show the typical values).
If ordinary movement triggers constantly, raise it.

Only when the model is final (no further tuning), evaluate the test session
**once**:

```bash
python src/diagnose_motion.py --checkpoint checkpoints/motion_v2_best.pt --split test
```

A second run is refused (`results/motion_v2_test_used.json`). Record the numbers
in NOTES.md.

## 3. Unblock the word dataset (GISLR), train, then live-test words mode (~20 signs, `--video`)

**Blocker found during the run:** every Kaggle download of `asl-signs` returns
`403 Forbidden`, even though listing the files works. That means the competition
rules haven't been accepted on this account. Mirrors weren't used (DECISIONS.md).

Everything downstream is built and tested against a synthetic dataset with the
same schema, so once the data arrives it's a sequence of commands, not code.

**3a. Accept the rules.** Log in to kaggle.com as the account whose token is in
`~/.kaggle/` (do item 5 first if you're rotating it), open
<https://www.kaggle.com/competitions/asl-signs/rules> and click *I Understand and
Accept*. Check it worked (this should download one small file, not 403):

```bash
kaggle competitions download -c asl-signs -f train.csv -p /tmp/gislr_check && ls -la /tmp/gislr_check
```

**3b. Download** (tens of GB; ~93 GB is free). Leave it as a zip, since the
preprocessor reads straight out of it (DECISIONS.md):

```bash
mkdir -p data/external/gislr logs
nohup kaggle competitions download -c asl-signs -p data/external/gislr \
    > logs/gislr_download.log 2>&1 &
tail -f logs/gislr_download.log          # Ctrl-C stops watching, not the download
```

**3c. Verify the schema, then build the cache.** `--verify-only` checks
train.csv, the sign map and the first parquet against what the code assumes
(columns, 543 landmarks per frame, NaN hands) and writes
`results/gislr_schema.json`. Read it before the full run.

```bash
python src/data/preprocess_gislr.py --verify-only
nohup python src/data/preprocess_gislr.py > logs/gislr_preprocess.log 2>&1 &
```

Success: `data/processed/gislr/` has one `.npy` + `.csv` per participant (21),
`index.csv` with ~94k rows, and `split.csv` with participant-disjoint
train/val/test. Rerunning skips finished shards.

**3d. Train** (about 10 runs is the budget). Each run writes a checkpoint,
history and `results/<run>_val_eval.json` with the one/two-handed and
static/dynamic breakdowns.

```bash
python src/train_words.py --model gru --features full --tag baseline 2>&1 | tee logs/words_gru_baseline.log
python src/train_words.py --model transformer --features full 2>&1 | tee logs/words_transformer_full.log
python src/ablation_words.py --epochs 30 2>&1 | tee logs/words_ablation.log     # 4 runs
```

`results/words_ablation.md` is the two-hand answer: `dom_shape` is what the
letters pipeline sees (one hand, no location), `both_shape` adds the other
hand, `location` adds where the hands are, `full` adds face and velocity. Look
at the two-handed column especially. Tune with the remaining 3–4 runs on
**val only** (`--dim 256`, `--epochs 60`, `--dropout 0.3`).

**3e. Test, once.** When the model is final:

```bash
python src/evaluate_words.py --checkpoint checkpoints/words_transformer_full_best.pt --split test
```

A second run is refused (`results/words_test_used.json`). Record val, test and
the ablation table in NOTES.md.

**3f. Live-test words mode on ~20 signs of your own.** Pick about 20 signs from
the 250 (names in `data/processed/gislr/index.csv`, column `sign`, or the
competition's `sign_to_prediction_index_map.json`). Mix one- and two-handed,
static and moving, using `results/words_sign_properties.csv`. Record 2–3 short
clips of each (webcam or phone, about 2 s, one sign per file, start and end
with hands down):

```
data/word_videos/<sign>/<anything>.mp4      e.g. data/word_videos/apple/ben_1.mp4
```

Then score them through exactly the live path, and watch a few in the demo:

```bash
python src/eval_word_videos.py --checkpoint checkpoints/words_transformer_full_best.pt
python src/demo.py --checkpoint checkpoints/mlp_pugeault_best.pt \
    --words-checkpoint checkpoints/words_transformer_full_best.pt \
    --video data/word_videos/apple/ben_1.mp4
python src/demo.py --checkpoint checkpoints/mlp_pugeault_best.pt \
    --words-checkpoint checkpoints/words_transformer_full_best.pt      # webcam; 'w' switches mode
```

`eval_word_videos.py` reports top-1/top-3 for the whole clip (model only) and
with automatic segmentation (model + segmenter), so you can see which one is
losing accuracy. Things that were only testable live:
- **Segmentation thresholds.** If a held final handshape doesn't end the sign,
  raise `--words-lull-speed` (default 0.4 shoulder-widths/s). If signs get cut
  mid-way, lower it. `t` (manual start/stop) always works as a fallback.
- **Mirroring.** Holistic runs on the un-flipped frame. If one-handed signs do
  noticeably worse than in val, compare against a left-handed attempt. The
  features canonicalise handedness, so they should score about the same.
- **FPS.** About 24 FPS measured with `--holistic-complexity 1` on your 640×480
  captures (`results/words_fps.json`). If the live loop feels slow, try
  `--holistic-complexity 0`.

## 4. Decide whether to push the branch

Nothing was pushed. The branch's history (inherited from `main`) still contains
the Kaggle API token that was once committed in `NOTES.md`, so pushing it
anywhere public republishes that token. See item 5 first.

```bash
git log --oneline main..feature/motion-v2-and-words   # what this run added
```

## 5. Security: confirm the leaked Kaggle token was rotated

A Kaggle API token was committed to `NOTES.md` in this repo's history. It has
never been confirmed as rotated. On kaggle.com → Settings → API, expire the old
token and create a new one, then put the new one in `~/.kaggle/access_token`
(and `~/.kaggle/kaggle.json` if you use it), with `chmod 600`. Check with:

```bash
kaggle competitions files -c asl-signs | head -3
```

Rewriting git history to remove the token is optional once it's rotated: a
rotated token is useless. It's your call either way; the autonomous run didn't
touch history.
