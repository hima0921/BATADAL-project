# Ablation study: configurations and runner

How `make_configs.py` and `run_ablations.py` set up and run the ablation study of the Transformer reconstruction model.

## 1. Purpose and files

An ablation study changes one setting of the model at a time and measures what happens. Here it answers one question: which settings affect how well the model detects attacks, and by how much.

| File | What it does |
|---|---|
| `make_configs.py` | Writes one JSON config file per configuration |
| `run_ablations.py` | Trains, calibrates and evaluates every configuration and seed, then collects the results and draws the plots |
| `plot_ablations.py` | Draws the plots. Called by the runner; run by hand only to redraw |
| `train_batadal_recon_model.py`, `calibrate_recon_model.py`, `evaluate_recon_model.py` | The three stages the runner calls. They must sit in the same folder as the runner |

The study reads Dataset 1 and Dataset 2 from `data/processed/`. The Test set is never read.

## 2. Summary

1. `make_configs.py` writes 12 config files: a baseline and 11 variations.
2. `run_ablations.py` runs each config with 3 seeds, 36 runs in total.
3. Each run is trained, calibrated and evaluated on its own, in its own folder.
4. All results are collected in one file, `results.csv`, and plotted.

```bash
python make_configs.py --out-dir configs/ablation
python run_ablations.py \
    --configs-dir configs/ablation \
    --dataset-path ../../data/processed \
    --runs-dir runs/ablation \
    --seeds 0 1 2 --keep-only-best
```

## 3. `make_configs.py`

### Design: one setting at a time

- **Every configuration differs from the baseline in exactly one setting.** A difference in the results then has one possible cause.
- **The list is fixed before anything is run.** Settings are edited in the script, not chosen after seeing results.
- **Limit of this design.** It does not show how settings interact. For example, it cannot say whether 8 latent tokens help more with a 12-hour window than with a 24-hour one.

### Baseline

The reference model, set in `BASELINE`. The keys are the same as in `24_batadal_recon_config.json`.

| Setting | Value |
|---|---|
| `max_global_steps` | 15,000 |
| `model_lr` | 3e-3 |
| `window` | 24 |
| `dropout` | 0.1 |
| `num_heads` | 8 |
| `num_enc_blocks` | 2 |
| `num_dec_blocks` | 2 |
| `hidden_dim` | 128 |
| `embedding_dim` | 64 |
| `num_latent_tokens` | 4 |
| `latent_dim` | 16 |
| `activation_type` | gelu |

### Settings that are varied

Set in `ABLATIONS`. The baseline's own value is skipped, because the baseline run already covers it.

| Setting | Values tried | What it controls | New configs |
|---|---|---|---|
| `window` | 12, **24** | Hours of history the model reads | 1 |
| `num_latent_tokens` | 1, 2, **4**, 8 | Number of tokens in the bottleneck | 3 |
| `latent_dim` | 4, 8, **16**, 32 | Values per bottleneck token | 3 |
| `num_enc_blocks` | 1, **2**, 3 | Depth of the encoder | 2 |
| `num_dec_blocks` | 1, **2**, 3 | Depth of the decoder | 2 |

Bold marks the baseline. With the baseline itself this gives 12 config files.

### Output

One file per configuration in the output folder:

- `baseline.json`
- `<setting>-<value>.json`, for example `latent_dim-8.json` or `window-12.json`

The file name is the configuration's name everywhere else: in the run folders, in `results.csv` and in the plots.

### Checks

The script stops with an error before writing a config that cannot run.

| Check | Reason |
|---|---|
| `embedding_dim` divisible by `num_heads` | Required by multi-head attention |
| `latent_dim` at least 4 | The decoder applies layer normalization to each latent token, which removes its mean and scale. A token of 1 value then carries nothing and a token of 2 carries one bit |
| `num_latent_tokens` at least 1 | The bottleneck needs a token |
| `window` is 12 or 24 | The data loader accepts only these. The 24-hour gaps between the data blocks were sized for them |
| At least one encoder and one decoder block | |
| `max_global_steps` positive | |

### Options

| Option | Meaning |
|---|---|
| `--out-dir` | Folder for the config files (required) |
| `--max-global-steps` | Overrides the training length in every config, for a shorter study |

## 4. `run_ablations.py`

### Three stages per run

Every configuration and seed gets its own folder, `<runs-dir>/<config>/seed<k>/`, and goes through three stages.

| Stage | Script | Data | Result |
|---|---|---|---|
| `train` | `train_batadal_recon_model.py` | Train and validation blocks of Dataset 1 | `best_model.pt`, `recon_loss.json` |
| `calibrate` | `calibrate_recon_model.py` | Calibration and validation blocks of Dataset 1 | `calibration/calibration.pt`, `calibration.json` |
| `evaluate` | `evaluate_recon_model.py` | Dataset 2 (`dataset2_eval.csv`) | `evaluation/metrics.json` and the evaluation plots |

**Each run has its own threshold.** A model that reconstructs normal data better has smaller errors and needs a lower threshold. A threshold shared between runs would favour whichever model it happened to suit. Every run is therefore calibrated on its own model, under the same false-alarm budget, and evaluated with its own frozen `calibration.pt`.

**The same settings for every run.** The budget, the number of top sensors, the episode rule and the grace hours are the same for all runs, so the only thing that differs between two runs is the configuration and the seed.

### Training details set by the runner

- **Learning-rate schedule.** `--lr-steps` is set to a quarter of `max_global_steps`. With 15,000 steps the rate is halved at steps 3,750, 7,500 and 11,250, and ends at one eighth of its starting value. The schedule scales with the run length, so shorter studies keep the same shape.
- **Seed.** Passed to the training script, where it sets the starting weights and the batch order.
- **Checkpoints.** Validated and saved every 2,500 steps by default. With `--keep-only-best`, the numbered checkpoints are deleted after training and only `best_model.pt` stays.

### Stopping and restarting

The runner can be stopped and started again. A stage that already finished is skipped.

| Stage | Counts as finished when |
|---|---|
| `train` | `train_done.txt` exists in the run folder |
| `calibrate` | `calibration.json` exists and is newer than `best_model.pt` |
| `evaluate` | `metrics.json` exists and is newer than `best_model.pt` and `calibration.pt` |

- **Results follow what they were made from.** If a model is retrained, its calibration and evaluation are redone. If a calibration is redone, its evaluation is redone.
- **A training run that was interrupted starts again from step 0.** The marker file is written only when training finishes.
- **`--force` redoes the selected stages** whether or not results exist.
- **A failed stage does not stop the study.** The later stages of that run are skipped, the other runs continue, and every failure is listed at the end.

### Collected results

After the last run, the runner reads three files from every run folder and writes one row per run to `<runs-dir>/results.csv`.

| Columns | From |
|---|---|
| `config`, `seed` | The folder names |
| `best_val_loss`, `best_step` | `recon_loss.json` |
| Thresholds and false-alarm figures on clean data, per score | `calibration/calibration.json` |
| Every Dataset 2 metric, as `<score>_<metric>` | `evaluation/metrics.json` |

A run that has not finished a stage still gets a row, with those columns empty.

### Plots

The runner then calls `plot_ablations.py`, which reads `results.csv` and writes to `<runs-dir>/plots/`:

- one figure per headline metric, with one panel per setting, every seed as a dot and the mean as a line
- validation loss during training, per setting and per configuration
- validation loss against PR-AUC, one point per configuration
- false-alarm episodes per week against recall, with the budget marked
- `ablation_summary.csv`, the mean, lowest and highest value across seeds for each configuration

A problem while plotting is reported but does not affect `results.csv`.

### Options

| Option | Default | Meaning |
|---|---|---|
| `--configs-dir` | required | Folder of config files from `make_configs.py` |
| `--dataset-path` | required | The `data/processed` folder |
| `--runs-dir` | required | Folder for the run folders, `results.csv` and plots |
| `--seeds` | 0 1 2 | Seeds run for every config |
| `--stages` | all three | Stages to run, for example `--stages calibrate evaluate` |
| `--device` | cpu | `cpu` or `cuda` |
| `--dataset2-path` | `dataset2_eval.csv` in `--dataset-path` | Dataset 2 with labels, if it is somewhere else |
| `--checkpoint-steps` | 2,500 | Steps between validations during training |
| `--top-k` | the script's own (3) | Top sensors averaged in the score, for every run |
| `--merge-gap-hours` | the script's own (0) | Episode rule, for every run |
| `--grace-hours` | the script's own (0) | Hours after each attack left out of the figures |
| `--keep-only-best` | off | Delete numbered checkpoints after training |
| `--force` | off | Redo the selected stages even if results exist |
| `--no-plots` | off | Skip the plots |
| `--plot-metrics` | headline metrics | Columns of `results.csv` to plot, or `all` |
| `--plot-score` | `topk_percentile` | The score the default plots show |
| `--dry-run` | off | Print the commands without running them |

### Common uses

```bash
# See every command first, without running anything.
python run_ablations.py --configs-dir configs/ablation \
    --dataset-path ../../data/processed --runs-dir runs/ablation --dry-run

# Redo calibration and evaluation with a different episode rule.
python run_ablations.py --configs-dir configs/ablation \
    --dataset-path ../../data/processed --runs-dir runs/ablation \
    --stages calibrate evaluate --merge-gap-hours 2 --force

# A short trial of the whole study.
python make_configs.py --out-dir configs/trial --max-global-steps 2000
python run_ablations.py --configs-dir configs/trial \
    --dataset-path ../../data/processed --runs-dir runs/trial --seeds 0
```

## 5. Reading the results

- **Compare on Dataset 2 metrics, not on validation loss.** A larger bottleneck always reconstructs normal data better. That does not mean it detects attacks better, because it may reconstruct attacks well too.
- **PR-AUC is the main figure.** It does not depend on the threshold. Attacks detected, detection delay and false-alarm episodes per week are read next to it.
- **Use the spread across seeds.** A difference between two configurations that is smaller than the difference between seeds of one configuration is not a finding.

## 6. Limits and open points

1. **Changed calibration settings need `--force`.** Passing a new `--top-k`, `--merge-gap-hours` or `--grace-hours` does not mark old results as out of date. Without `--force`, finished runs keep their old settings and the study mixes two.
2. **The baseline learning rate is 3e-3.** Earlier runs used 3e-4. The value should be confirmed before a full study, because every configuration inherits it.
3. **Choosing a configuration on Dataset 2 uses up Dataset 2.** The chosen configuration's Dataset 2 figures are then slightly optimistic. The Test set stays untouched until the choice is frozen.
4. **Only 7 attacks.** "Attacks detected" will often tie between configurations.
5. **Runs from before the key rename cannot be reused.** Checkpoints saved with `num_encodings` and `bottleneck_dim`, or trained on the earlier data loader, should be moved out of the runs folder first. Otherwise training is skipped for them and the later stages fail.
6. **Cost.** 36 runs of 15,000 steps each. The trial command in section 4 is a quick way to confirm that everything works before starting.
