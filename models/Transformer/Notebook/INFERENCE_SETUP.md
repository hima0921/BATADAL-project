# Inference notebook: setup and run

How to install and run `transformer_inference.ipynb`, which scores Dataset 2 with the frozen 12-hour baseline Transformer (the best model from the ablation study) and its frozen calibration, and summarizes the result. Nothing is trained or recalibrated.

## 1. Requirements

- Python 3.13 or later (the notebook was run on 3.14) and the project's `requirements.txt`
- Jupyter: VS Code with the **Python** and **Jupyter** extensions, or JupyterLab
- The processed data in `data/processed/`, from `notebooks/Pipeline.ipynb`

## 2. Files

The notebook sits in its own folder under `models/Transformer/`, next to a `Model Weights/` folder:

```
<repo>/
├── requirements.txt
├── data/
│   └── processed/                  train_scaled.csv, val_scaled.csv, cal_scaled.csv,
│                                   dataset2_eval.csv, scaler.joblib
└── models/Transformer/
    ├── model.py, layers.py, batadal_dataset.py, evaluate_recon_model.py
    └── <notebook folder>/
        ├── transformer_inference.ipynb
        └── Model Weights/
            ├── best_model.pt
            └── calibration/calibration.pt
```

| File | What it is |
|---|---|
| `best_model.pt` | The 12-hour baseline, seed 0, trained to step 25,000 (2.4 MB) |
| `calibration.pt` | Its frozen calibration: top 3 sensors, 2-hour merge gap, budget of 1 episode per week and 1% of hours |
| `model.py`, `layers.py`, `batadal_dataset.py`, `evaluate_recon_model.py` | The project code the notebook imports from `CODE_DIR` |

## 3. Install

From the repository root:

```bash
python3 -m venv myenv
source myenv/bin/activate          # Windows: myenv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install ipykernel
python -m ipykernel install --user --name myenv --display-name "Python (myenv)"
```

The last line registers the environment as a Jupyter kernel, so the notebook can use it.

## 4. Open the notebook on the right kernel

**VS Code**

1. Open `transformer_inference.ipynb`.
2. Click the kernel name at the top right → **Select Another Kernel** → **Jupyter Kernel** → **Python (myenv)**.
3. If it is not listed, run **Developer: Reload Window** from the Command Palette (Ctrl+Shift+P) and try again.

**JupyterLab**

```bash
source myenv/bin/activate
python -m pip install jupyterlab
jupyter lab
```

Open the notebook, then **Kernel → Change Kernel → Python (myenv)**.

**Check the kernel** in a new cell before running anything:

```python
import sys, torch
print(sys.executable)      # should end in myenv/bin/python
print(torch.__version__)
```

## 5. Settings

The settings cell at the top of the notebook, with the paths for the layout in section 2:

```python
CODE_DIR = Path("..")                                     # models/Transformer
DATA_DIR = Path("../../../data/processed")                # repo root / data / processed
SOURCE_CHECKPOINT = Path("Model Weights/best_model.pt")   # the full checkpoint
CHECKPOINT = Path("Model Weights/best_model.pt")          # the file scored with (a new name makes a slim copy in step 1)
CALIBRATION = Path("Model Weights/calibration/calibration.pt")
OUT_DIR = Path("inference_results")

SCORE = "topk_percentile"
GRACE_HOURS = 3
BATCH_SIZE = 128
DEVICE = "cpu"
```

- **Every path must be a `Path(...)`,** not a plain string.
- **Paths are relative to the notebook's folder,** not the repository root.
- **`GRACE_HOURS` is not stored in `calibration.pt`.** Keep it at 3, the value the calibration was evaluated with.
- **`DEVICE = "cuda"`** uses a GPU if one is available; the results are the same.
- **To check the paths,** run:
  ```python
  for p in [CODE_DIR / "model.py", DATA_DIR / "train_scaled.csv", CHECKPOINT, CALIBRATION]:
      print(p, p.exists())
  ```

## 6. Run

Run the cells from top to bottom.

| Step | What it does |
|---|---|
| 1. Slim checkpoint | Optional. With `CHECKPOINT` equal to `SOURCE_CHECKPOINT`, it does nothing and the full file is used. To make a smaller copy without the optimizer state, give `CHECKPOINT` a new name, such as `Model Weights/model_w12_seed0.pt` |
| 2. Load | Imports the project code, builds the model from the checkpoint and loads the calibration, checking that they match |
| 3. Data | Loads the processed files: the three Dataset 1 blocks and Dataset 2 |
| 4. Score Dataset 2 | Scores every hour from the 12-hour window that ends at it, applies the frozen thresholds, and writes the evaluation files and plots |
| 5. Summary | Summarizes the scores: headline figures, hour counts, each attack, each false alarm, the score distribution and the evaluation plots, then saves the summary files |

**What a correct run prints.** Step 2:

```
Window: 12 h | latent: 4 tokens x 16 | parameters: 192,820 | device: cpu
Checkpoint step: 25,000 | seed: 0
Calibration: top-k 3, merge gap 2 h, budget 1 episode/week and 1% of hours
Thresholds: {'topk_percentile': 0.998448, 'mean': 0.001791, 'topk_ratio': 32.761668}
```

Step 5's headline, for the `topk_percentile` score: 7 of 7 attacks detected, mean delay 1.7 hours, PR-AUC 0.792; 1.15 false-alarm episodes per week and 0.88% of clean hours, outside the budget of 1 episode per week. Its hour counts total 4,145: 492 attack hours and 3,653 clean hours, with the 21 grace hours left out. Different figures mean a different checkpoint, calibration, data or `GRACE_HOURS`.

## 7. Reading the summary

| Part | What it shows |
|---|---|
| Headline | Two sentences for `SCORE`, then every metric for all three scores. Each score has its own frozen threshold; only `SCORE` is the one reported |
| Hours | Attack and clean hours, split by alarm and no alarm. The grace hours after each attack are counted in neither |
| Each attack | Start, length, delay from the real start, share of the attack's hours under alarm, peak score, and the three sensors with the largest errors |
| False alarms | Every alarm episode outside the attack windows, with its peak score, the hours until the next attack starts, and the sensors behind it. An episode ending shortly before an attack may be its first effect rather than a false alarm |
| Score distribution | Clean and attack hours against the threshold |
| Evaluation plots | The score timeline, the delay per attack and the main sensors per attack |

**Sensor contributions.** Each sensor is shown as a multiple of its normal 99th-percentile error on the calibration block: 1x is the edge of its normal range, 5x is five times beyond it.

**The percentile score on the distribution plot.** The score cannot exceed 1, so it is drawn as its distance from 1 on a log scale: 1 on the axis is a score of 0.9, 2 is 0.99, 3 is 0.999, and hours at exactly 1 are drawn at 6. Many attack hours sit at that ceiling, where the score can no longer rank them against each other.

## 8. Outputs

Written to `OUT_DIR`:

| File | Contents |
|---|---|
| `metrics.json` | All Dataset 2 metrics, as `<score>_<metric>` |
| `per_attack.csv` | Detected or missed, and the delay, for each attack |
| `hourly_scores.csv` | Score and alarm for every scored hour |
| `timeline_<score>.png`, `attack_<n>.png`, `false_alarm_<n>.png`, `detection_delay.png`, `top_sensors_per_attack.png` | Evaluation plots |
| `score_distribution_<score>.png` | Clean and attack hours against the threshold |
| `dataset2_summary.json` | The headline figures and hour counts |
| `attack_summary.csv` | The per-attack table |
| `false_alarm_summary.csv` | The false-alarm table |

The first 11 hours of Dataset 2 have no score: no 12-hour window ends there.
