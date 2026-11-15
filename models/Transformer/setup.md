# Setup and run

How to install the project and run the Transformer reconstruction model.

## 1. Requirements

- Python 3.13
- The three BATADAL files in a `data/` folder:
  - `Training dataset 1.csv`
  - `Training dataset 2.csv`
  - `BATADAL_test_dataset.csv`

## 2. Install

Run these from the repository root.

```bash
python3.13 -m venv <Your env file>
source myenv/bin/activate          # Windows: myenv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pins five packages: `torch`, `numpy`, `pandas`, `matplotlib` and `scikit-learn`.

If you have an NVIDIA GPU and want to use it, install `torch` first with the command for your CUDA version from https://pytorch.org, then run `pip install -r requirements.txt`.

## 3. Run

Run the scripts from the folder that contains `model.py`, `layers.py` and `batadal_dataset.py`. Change the `data/` paths if your data folder is somewhere else. The quotes are needed because the file names contain spaces.

### Step 1: Check the data loader

```bash
python batadal_dataset.py "data/Training dataset 1.csv" "data/Training dataset 2.csv"
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
    --dataset-path "data/Training dataset 1.csv" \
    -c 24_batadal_recon_config.json \
    --out-dir "runs/recon_window_24" \
    --checkpoint-steps 250 \
    --seed 69
```

- Add `--device cuda` to train on a GPU.
- Use `-c 12_batadal_recon_config.json` for the 12-hour window.
- To resume a run, add `--model-checkpoint runs/recon_seed0/<steps>_model.pt --load-optim`.

Outputs in `runs/recon_window_24/`:

| File | What it is |
|---|---|
| `best_model.pt` | The checkpoint with the lowest validation loss |
| `<steps>_model.pt` | A checkpoint at every checkpoint step |
| `recon_loss.png` | Train and validation loss curves |
| `BATADAL Recon Model.log` | The training log |


## 4. Common problems

| Problem | Fix |
|---|---|
| `No module named 'torch'` | The virtual environment is not active. Run `source myenv/bin/activate` |
| `No such file or directory` for a CSV | Check the path and keep the quotes around it |
| `Test set is locked` | Expected. The Test set stays unused until all choices are frozen |

## 5. More detail

- `docs/data_and_training.md`: the reason for each choice in the loader and training code
- `docs/audit.md` and `docs/exploration.md`: the data reports
