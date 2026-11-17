# Setup and run

How to install the project and run the Transformer reconstruction model.

## 1. Requirements

- Python 3
- The raw BATADAL files in the repository's `data/` folder:
  - `Training dataset 1.csv`
  - `Training dataset 2.csv`
  - `BATADAL_test_dataset.csv`
- The processed files in `data/processed/` (step 3.0 creates them)

## 2. Install

Run these from the repository root.

```bash
python3 -m venv myenv
source myenv/bin/activate          # Windows: myenv\Scripts\activate
python -m pip install -r requirements.txt
```

`requirements.txt` pins six packages: `torch`, `numpy`, `pandas`, `matplotlib`, `joblib` and `scikit-learn`. To run the notebooks you also need Jupyter (`python -m pip install notebook`).

If you have an NVIDIA GPU and want to use it, install `torch` first with the command for your CUDA version from https://pytorch.org, then install the rest.

Every new terminal needs `source myenv/bin/activate` again. If Python reports `No module named 'torch'`, the environment is not active.

## 3. Run

The scripts are in `models/Transformer/`. Run them from that folder:

```bash
cd models/Transformer
```

All commands below use `../../data/processed`, which is the processed data folder seen from there.

### Step 0: Create the processed data (once)

Run `notebooks/Pipeline.ipynb` from top to bottom. It rounds the data, drops the 7 columns that never change, splits Dataset 1, scales, and saves five files in `data/processed/`:

- `train_scaled.csv`, `val_scaled.csv`, `cal_scaled.csv`
- `dataset2_scaled.csv`
- `scaler.joblib`

Skip this step if the folder already has them. `docs/pipeline.md` describes each step.

### Step 1: Check the data loader

```bash
python batadal_dataset.py ../../data/processed ../../data/processed/dataset2_scaled.csv
```

You should see 36 features and these window counts:

| Loader | Windows |
|---|---|
| train | 5,233 |
| val | 1,705 |
| calib | 1,706 |
| dev | 4,154 |

### Step 2: Train the model

```bash
python train_batadal_recon_model.py \
    --dataset-path ../../data/processed \
    -c 24_batadal_recon_config.json \
    --out-dir runs/recon_24h \
    --seed 0
```

- `--dataset-path` is the processed data **folder**, not a CSV file.
- Use `-c 12_batadal_recon_config.json` and another `--out-dir` for the 12-hour window.
- Add `--device cuda` to train on a GPU.
- To resume a run, add `--model-checkpoint runs/recon_24h/<steps>_model.pt --load-optim`.

A run is 15,000 steps by default and saves a checkpoint every 1,000 steps. Outputs in `runs/recon_24h/`:

| File | What it is |
|---|---|
| `best_model.pt` | The checkpoint with the lowest validation loss |
| `<steps>_model.pt` | A checkpoint at every checkpoint step |
| `recon_loss.png` | Train and validation loss curves |
| `BATADAL Recon Model.log` | The training log |

### Step 3: Calibrate the alarm threshold

```bash
python calibrate_recon_model.py \
    --dataset-path ../../data/processed \
    --model-checkpoint runs/recon_24h/best_model.pt \
    --out-dir runs/recon_24h/calibration
```

This sets the threshold on clean Dataset 1 data and saves it in `calibration.pt`, with a summary in `calibration.json` and one plot per score.

### Step 4: Evaluate on Dataset 2

```bash
python evaluate_recon_model.py \
    --dataset-path ../../data/processed \
    --dataset2-path ../../data/processed/dataset2_scaled.csv \
    --model-checkpoint runs/recon_24h/best_model.pt \
    --calibration runs/recon_24h/calibration/calibration.pt \
    --out-dir runs/recon_24h/evaluation
```

Outputs in `runs/recon_24h/evaluation/`:

| File | What it is |
|---|---|
| `metrics.json`, the log | Attacks detected, delay, false alarms, precision, recall, F1, AUCs and the S-score |
| `per_attack.csv` | Detected or missed, and the delay, for each of the 7 attacks |
| `hourly_scores.csv` | The score and alarm for every hour |
| `timeline_*.png` | The score over time with the threshold and attack windows |
| `attack_*.png`, `false_alarm_*.png` | Close-ups with the sensors that contributed most |
| `sensor_errors.npz` | Per-sensor errors, for combining with other detectors |

The Test set is not used by any of these steps.

### Three seeds

The proposal calls for three seeds. Repeat steps 2 to 4 with `--seed 1` and `--seed 2`, and a separate `--out-dir` for each.

### Ablation study (optional)

```bash
python make_configs.py --out-dir configs/ablation
python run_ablations.py \
    --configs-dir configs/ablation \
    --dataset-path ../../data/processed \
    --dataset2-path ../../data/processed/dataset2_scaled.csv \
    --runs-dir runs/ablation \
    --seeds 0 1 2 --keep-only-best
```

This trains, calibrates and evaluates every configuration and seed, then writes `runs/ablation/results.csv` and the plots in `runs/ablation/plots/`. Add `--dry-run` first to see the commands without running them.

## 4. Common problems

| Problem | Fix |
|---|---|
| `No module named 'torch'` | The virtual environment is not active. Run `source myenv/bin/activate` |
| `No module named pip` | Run `python -m ensurepip --upgrade` |
| `No space left on device` during install | Run `python -m pip cache purge`, or install the CPU-only build of `torch` |
| `expected the folder of processed files` | `--dataset-path` was given a CSV file. Pass the `data/processed` folder |
| `does not look like a processed file` | A raw CSV was passed. Use the files in `data/processed/` |
| `No such file or directory` for a processed file | Run `notebooks/Pipeline.ipynb` first (step 0) |
| `unexpected keyword argument` from `build_dataloaders` | The scripts and `batadal_dataset.py` are from different versions. Pull the latest of all of them |
| `Scaler differs from the checkpoint's` | The checkpoint was trained on different data than the folder you passed |
| `gap between the ... blocks is 24 hours, shorter than the window` | The window is longer than 24 hours, which the processed files do not allow |
| `Test set is locked` | Expected. The Test set stays unused until all choices are frozen |

## 5. More detail

- `docs/pipeline.md`: the shared data preparation
- `docs/data_and_training.md`: the reason for each choice in the loader and training code
- `docs/audit.md` and `docs/exploration.md`: the data reports
- `docs/ablation_observations.md`: what the first ablation study showed
