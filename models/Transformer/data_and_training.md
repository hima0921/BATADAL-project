# Data loader and training code: design choices

## 1. Purpose and files

This document describes the data loader and the training script for the Transformer reconstruction model, and gives the reason for each choice. A reason comes from one of three places: our own data reports (`docs/audit.md`, `docs/exploration.md`, `docs/pipeline.md`), the project proposal, or published work. Published work is cited as [n] and listed in section 8.

The data is prepared once, by the shared pipeline, and saved in `data/processed/`. The loader reads those files. It does not round, drop columns, split or scale anything itself.

| File | What it does |
|---|---|
| `notebooks/Pipeline.ipynb` | The shared data preparation: rounds, drops the 7 never-changing columns, splits Dataset 1, scales, and saves the results. Described in `docs/pipeline.md` |
| `data/processed/` | The saved results: `train_scaled.csv`, `val_scaled.csv`, `cal_scaled.csv`, `dataset2_scaled.csv`, `scaler.joblib` |
| `batadal_dataset.py` | Reads and checks the processed files, builds labels, cuts windows, returns the DataLoaders |
| `train_batadal_recon_model.py` | Trains the model on the train block, validates on the validation block, saves checkpoints and a loss plot |
| `[12/24]_batadal_recon_config.json` | Model and training settings read by the training script |
| `model.py`, `layers.py` | The model. Covered here only where a training choice depends on it |

## 2. Summary table

| Loader | Source | Rows | Windows (24 h) | Shuffled | Used for |
|---|---|---|---|---|---|
| `train` | `train_scaled.csv` (Dataset 1, first 60%) | 5,256 | 5,233 | Yes | Fitting the model |
| `val` | `val_scaled.csv` (next 20% minus a 24 h gap) | 1,728 | 1,705 | No | Choosing the checkpoint |
| `calib` | `cal_scaled.csv` (last 20% minus a 24 h gap) | 1,729 | 1,706 | No | Setting alarm thresholds (not read during training) |
| `dev` | `dataset2_scaled.csv` (Dataset 2) | 4,177 | 4,154 | No | Development: label experiments, attack checks |
| `test` | Raw Test set CSV, prepared by the loader | 2,089 | 2,066 | No | Final evaluation only (locked) |

The row counts are from `docs/pipeline.md`. With the 12-hour window the counts are 5,245, 1,717 and 1,718 for the three Dataset 1 blocks.

| Setting | Default | Where set |
|---|---|---|
| Window | 24 hours (a 12-hour config also exists) | Config file |
| Split | 60 / 20 / 20, 24-hour gaps | `Pipeline.ipynb` |
| Scaling range | 0 to 1 | `Pipeline.ipynb` |
| Constant columns | Dropped (36 features) | `Pipeline.ipynb` |
| Rounding | 2 decimals | `Pipeline.ipynb` |
| Label rule for Dataset 2 | Real attack windows (`schedule`) | Loader |
| Bottleneck | 4 tokens of 16 values | Config file (`num_encodings`, `bottleneck_dim`) |
| Training length | 15,000 steps | Config file |
| Learning rate | 3e-3, halved every 3,500 steps (default) | Config and `--lr-steps`, `--lr-gamma` |
| Dropout | 0.1 | Config file |
| Batch size | 128 | `--batch-size` |
| Checkpoint interval | 1,000 steps | `--checkpoint-steps` |
| Seed | 0 | `--seed` |

## 3. Data loader

The first four choices below (rounding, constant columns, split, scaling) are made by the shared pipeline. The loader reads the result and checks it. They are described here because the model depends on them.

### Training on normal data only

- **The three Dataset 1 blocks must be entirely `ATT_FLAG = 0`, and the model is fitted on nothing else.** The loader stops with an error otherwise.
- Source: audit ("Labels"). Dataset 1 is all 0. Dataset 2 has no rows labeled 0.
- Literature: this is the standard set-up for reconstruction-based detection in water systems. Taormina and Galelli train autoencoders on normal operating data only, so that attacks are reproduced poorly [2]. Nemade et al. train their LSTM autoencoder only on attack-free sequences [4]. The review by Habib et al. finds unsupervised models the better fit when attack examples are scarce and normal data is plentiful [5].

### Loading checks

- **The loader only accepts the processed files.** A raw CSV, or a file that still has the 7 dropped columns, is rejected with a message saying to pass the files in `data/processed/`.
- **Rows must be strictly hourly inside each file.** The audit found no gaps or repeats ("Time"). The check is repeated on every load because a window across a gap is not 24 real hours.
- **All features are cast to float32.** The same column is int in one file and float in another (audit, "Data types").
- **Empty cells and bad on/off values stop the run.** The audit found none, so nothing is filled in.
- **The Training file must really be the scaled one.** Every continuous column in it has to run from 0 to 1, as `docs/pipeline.md` reports (step 3).
- **All files must have the same feature columns in the same order**, and the saved scaler must have been fitted on the same continuous columns.
- **The L_T6 limit at 5.5 and the negative F_PU7 value are left alone.** The audit did not treat them as errors ("Value ranges").

### Rounding

- **All files are rounded to 2 decimals, by the pipeline, before the split and the scaling.**
- Source: pipeline decision 1, from audit issue 5 and exploration step 4. Dataset 1 has up to 9 decimals and the other files have 2. `P_J280` moves over a range of 0.017, so at 2 decimals it has only three possible values. After scaling those are 0, 0.5 and 1, so one 0.01 change in the raw reading is a jump of 0.5 (pipeline step 3).
- Literature: none of the BATADAL studies we read mention this difference [2][3][4]. It is a choice made from our own audit.

### Scaling

- **Min-max scaling to the range 0 to 1 is fitted on the 28 continuous columns of the train block only. The 8 on/off columns are already 0 or 1 and are not scaled.**
- Source: proposal (Milestone 1) and pipeline steps 3 to 5. The loader reads the fitted scaler from `scaler.joblib` and never fits one itself. Asking the loader for another range raises an error.
- Literature: fitting the scaler on normal training data only is common practice. Nemade et al. fit theirs on normal rows only [4], and Salici et al. take their normalization statistics from nominal training data only [11]. Both of those use standard (z-score) scaling. We use min-max because the proposal specifies it.
- **Scaled values are not clipped.** `L_T1` and `P_J415` go above the training maximum in later files (audit, "Value ranges"), and a value outside the learned range is useful evidence. Validation and calibration stay close to 0 to 1, but Dataset 2 runs from about -1.28 to 2.86 after scaling (pipeline step 4). This differs from Salici et al., who clip to a fixed range [11].

### Constant columns

- **The 7 columns that never change in Dataset 1 or Dataset 2 are dropped, leaving 36 features: 28 continuous and 8 on/off.** They are not in the processed files. Asking the loader to keep them raises an error.
- Source: pipeline decision 2, from audit issue 1 and exploration decisions 1 and 2.
- Side effect: no model sees pump 3 running, or `S_PU1` at 0, both of which happen in the Test set (audit, "Value ranges").
- Literature: Nemade et al. train on all 43 columns [4], and the proposal also says 43. The difference from the proposal should be stated in the report (section 5).

### Split

- **Dataset 1 is split by time into 60% train, 20% validation and 20% calibration, with a 24-hour gap after each cut.** The split is fixed in the processed files. The loader works out the gap from the dates and raises an error if it is shorter than the window, so windows longer than 24 hours cannot be used.
- Source: proposal (Milestone 1) and pipeline step 2. The exploration's drift check found every sensor's average shifts by less than 0.1 SD across these blocks (step 1), so the cut points stay.
- Literature: the split is chronological because Transformers need care with small data sets. The ViT paper notes that Transformers lack built-in inductive biases and do poorly without enough data [9], and Kaplan et al. show that overfitting grows as the model gets large relative to the data [13]. A held-out validation block is how we see that happening.

### Windows

- **Each item is a 24-hour window of all features, and the target is the same window.** The window length is a setting, up to 24 hours.
- Source: proposal (12-hour and 24-hour windows). Exploration step 4 found tank 1 cycles about every 2 days. A longer window would need longer gaps in the pipeline.
- Literature: Nemade et al. also use 24-hour sliding windows for their LSTM autoencoder, and report that a sequence autoencoder separates attacks better than a dense one (ROC-AUC 0.976 against 0.968) [4].
- **A window is labeled by its last hour, and that hour's row number is returned.** The score for an hour then uses only that hour and earlier ones.
- Source: audit issue 4. Detection delay must be measured from the real attack start.
- Literature: time to detection is part of the official BATADAL score [1], so the hour an alarm is raised has to be well defined. Several Transformer detectors are evaluated with point adjustment, where one detected hour credits a whole attack; the Anomaly Transformer code does this [15]. The proposal rules it out.

### Labels for Dataset 2

- **`label_rule` chooses one of three rules: `file`, `schedule` (default) or `unlabeled_as_normal`.** Every hour also gets an attack ID from the 7 real attack windows.
- The processed Dataset 2 file keeps the raw `ATT_FLAG`. The pipeline has not built labels yet (its open item 1), so the loader still does.
- Source: audit issues 2, 3 and 4, and the proposal's label-sensitivity experiment. The file labels 219 attack hours, but the BATADAL site lists 492.
- Literature: Nemade et al. evaluate on Dataset 2 with the 219 labeled hours as attacks and the other 3,958 rows as normal [4]. That is our `unlabeled_as_normal` rule. By our audit it counts about 273 real attack hours as normal, so their reported scores are not directly comparable with ours.

### Test set

- **Loading the Test set needs `unlock_test=True`.** It has no labels (audit issue 6), and the audit says it stays untouched until all choices are frozen.
- **The pipeline does not save a scaled Test file, so the loader prepares the raw one.** It rounds to 2 decimals, drops the same 7 columns and scales with `scaler.joblib`, as `docs/pipeline.md` describes (step 7).

### What the loader does not do

- It does not round, drop columns, split or scale. `Pipeline.ipynb` does.
- It does not supply next-hour targets for a forecasting model. The GRU detector needs its own windowing (pipeline open item 3).
- It does not use BATADAL 2.0.

## 4. Training code

### Objective

- **The loss is the mean squared error between each window and its reconstruction.**
- Literature: reconstruction error from a model trained on normal data is the usual anomaly signal for autoencoders [2][6]. Salici et al. train their token-bottleneck Transformer the same way [11].

### What is trained

- **A Transformer autoencoder with four learnable bottleneck tokens and a decoder of 24 learned queries.** The settings come from the config file.
- Literature: learnable tokens that summarize a sequence go back to BERT's classification token [8]. Using several of them as an autoencoder bottleneck appears in Salici et al. for time series [11] and in TiTok for images [16]. Learned decoder queries with cross-attention come from DETR [10] and Perceiver IO [12]. Pintilie et al. show why a bottleneck matters: without one, a Transformer decoder can reconstruct too easily to learn a useful representation [14].
- **The bottleneck is 4 tokens of 16 values each, so a window is compressed to 64 numbers.** Both sizes are settings in the config file: `num_encodings` for the number of tokens and `bottleneck_dim` for the values per token. Both are saved in the checkpoint.
- **The output layer is a sigmoid, so every reconstructed value lies between 0 and 1.** This matches the scaling range. The bottleneck values pass through `tanh` and lie between -1 and 1; that range is internal to the model and unrelated to the data scaling.
- **GELU activation, layer normalization before each block, sinusoidal positions.** GELU follows BERT and ViT [8][9]. Normalizing before each block follows ViT [9]. The positional encoding is the original Transformer's [7].

### Optimizer and schedule

- **Adam with the library's default settings, learning rate 3e-4, halved every 3,500 steps over a 15,000-step run.** The rate ends at 3.75e-5.
- Source: the proof-of-concept run used a constant 1e-3 and its loss spiked repeatedly late in training. The lower, decaying rate is a response to that. An earlier default of 3,000 steps let the rate fall to almost nothing by the middle of a 15,000-step run, so it was raised.
- The 3,500 default assumes 15,000 steps. If `max_global_steps` changes, set `--lr-steps` to about a quarter of it.
- Literature: Adam is the optimizer of the original Transformer, which pairs it with a warm-up period before decaying the rate [7]. Our script has no warm-up. Salici et al. use Adam at 1e-3 for 50 epochs [11].
- **Gradients are clipped at norm 1.** ViT reports this as helpful for training Transformers [9].

### Regularization

- **Dropout of 0.1 in the example config.** The original Transformer's base model and BERT both use 0.1 [7][8].
- Source: the proof of concept ran with no dropout and showed a four-fold gap between train and validation error.
- Literature: the gap is the expected risk. Our model has roughly as many parameters as the training block has distinct values, the regime where Kaplan et al. find overfitting sets in [13]. The script's weight decay is zero.

### Validation and model selection

- **Every `--checkpoint-steps` steps, the script measures loss over the whole train block and the whole validation block in evaluation mode.** Both are measured the same way so the two curves are comparable.
- **The checkpoint with the lowest validation loss is copied to `best_model.pt`.** Training always runs to `max_global_steps`; there is no early stopping.
- Literature: Kaplan et al. treat stopping at the best validation point as the standard remedy when data is limited [13]. Keeping the best checkpoint has the same effect.
- **Limit of this selection.** Validation data is all normal, so validation loss measures reconstruction quality, not detection. MEMTO was designed around the problem that reconstruction models can generalize so well that they reconstruct anomalies too [17]. The lowest validation loss is therefore not proof of the best detector. That has to be checked on Dataset 2.

### Checkpoints and outputs

| Output | Contents |
|---|---|
| `<steps>_model.pt` | Weights, optimizer state, step count, model settings, feature names, scaler values, loss history |
| `best_model.pt` | Copy of the checkpoint with the lowest validation loss |
| `recon_loss.png`, `recon_loss.json` | Train and validation loss at each checkpoint, best step marked |
| `BATADAL Recon Model.log` | Settings and per-step loss |

- **The scaler values and feature names are stored in every checkpoint.** Scoring new data needs the same scaling, and the calibration and evaluation scripts check for a match. Checkpoints trained with the earlier raw-file loader store values for all 43 raw columns; the check compares them on the 36 kept columns, so those checkpoints still load.
- **`--model-checkpoint` with `--load-optim` resumes a run.** The learning-rate schedule restarts from the config value.
- **`--seed` sets the model's starting weights and the batch order.** The default is 69. The proposal calls for three seeds.

### What the training script never reads

The calibration block, Dataset 2 and the Test set. Thresholds and evaluation come after the model is frozen.

## 5. Issues and decisions we need to make

1. **Output layer and out-of-range values.** The data is scaled to 0 to 1 and the model ends in a sigmoid, as the proposal specifies. Two consequences remain. Because values are not clipped, some scaled values fall outside 0 to 1, which a sigmoid cannot produce, so those readings always carry some reconstruction error. In Dataset 2 this is large: one value at 2.86 has a squared error of at least 3.5 on its own. And the on/off features sit exactly at 0 or 1, where a sigmoid trained with squared error learns slowly.
2. **Differences from the proposal.** The proposal uses 43 features. The shared pipeline uses 36 (decision 2). The report should say so.
3. **Rounding.** No published BATADAL study we read does this. It follows from our audit (pipeline decision 1) and should be stated in the report. `P_J280` becomes a three-value signal whose steps are 0.5 after scaling.
4. **Label rule.** The default treats the 3,685 hours outside the attack windows as normal. They are unlabeled in the file. The pipeline's plan for a shared labels file (its open item 1) is not done; until then the loader's labels are the only version.
5. **Models trained before the switch to the shared data.** The earlier loader read the raw CSVs and rounded them itself, after converting to float32. A value such as 2.985000002 then rounded to 2.98 where the pipeline gives 2.99. In a test on made-up data this changed 11 of about 250,000 Dataset 1 cells; Dataset 2 was not affected. The count on the real data is not known. Existing checkpoints were therefore trained on very slightly different values. Decision needed: accept them, or retrain on the processed files.
6. **No warm-up and no weight decay.** The original Transformer uses warm-up [7]. Whether adding either helps here is untested.
7. **Model selection by reconstruction loss.** See the limit in section 4. A check on Dataset 2 is needed before a checkpoint is treated as final.
8. **Small `bottleneck_dim` values do not work as intended.** The decoder applies layer normalization to each bottleneck token, which removes its mean and scale. A token of 1 value then carries nothing and a token of 2 carries one bit. In general a token carries about two fewer free values than its size. Keep `bottleneck_dim` at 4 or above.
9. **Windows longer than 24 hours.** The gaps in the processed files are 24 hours, so a 48-hour window is rejected. Trying one needs a change in `Pipeline.ipynb`.
10. **Comparability with our earlier runs.** The proof of concept scaled the data to -1 to 1 with a `tanh` output. Halving the range divides squared error by four, so losses from the current code are about four times smaller for the same reconstruction quality. Loss values from before and after the change cannot be compared directly.
11. **Comparability with published BATADAL results.** Scores in [3] and [4] use different label handling and thresholds chosen with knowledge of the attacks. Nemade et al. pick the threshold that maximizes F1 on the labeled series, and train their autoencoder on the same series it is scored on [4]. Ours is trained on Dataset 1 only and its threshold comes from normal data under a false-alarm budget, so lower scores are expected.

## 6. To do

1. **Run the loader and one training run on the real `data/processed/` folder**, and confirm the window counts in section 2.
2. **Finish the bottleneck ablation on Dataset 2.** The bottleneck has two sizes that can be varied independently: the number of tokens and the values per token.
   - Done: both are config settings, and a first study trained 12 configurations with three seeds each. Its observations on reconstruction are in `docs/ablation_observations.md`.
   - Still to do: compare the configurations on Dataset 2, by how well the scores separate attack hours from normal hours. Validation loss cannot be used for this, because a larger bottleneck always reconstructs normal data better.
   - Why it matters: the bottleneck sets the trade-off the model depends on. Too large and it can reconstruct attacks as well as normal data, the over-generalization problem MEMTO addresses [17]. Too small and normal data is reconstructed poorly, which raises false alarms.
   - Literature: published token-bottleneck models use different sizes and treat the size as a design choice. Salici et al. use 2 tokens [11], and TiTok reports variants with 32, 64 and 128 tokens [16].
   - Scope: this ablation is not in the proposal. It is an extra, after the required comparisons.
3. **Decide on the checkpoints trained before the switch** (section 5, item 5).

## 7. What was not checked

- The loader on the real `data/processed/` files. It was checked on made-up data only.
- Training with the new loader. Every run so far used the earlier raw-file loader.
- How many Dataset 1 cells differ between the earlier loader and the processed files on the real data (section 5, item 5).
- Whether the lower learning rate and dropout fix the loss spikes and the train-validation gap seen in the proof of concept.
- Resuming from a checkpoint.
- Detection on Dataset 2. That is the job of `evaluate_recon_model.py` and is outside this document.
- The literature list is not a complete survey. The Transformer detectors in [11], [14], [15], [17] and [18] were found in short searches.
- Reference [2] was read from its abstract only, because our copy is a scanned manuscript.

## 8. References

Papers the team provided:

1. Taormina, R., et al. (2018). The battle of the attack detection algorithms: Disclosing cyber attacks on water distribution networks. *Journal of Water Resources Planning and Management*, 144(8).
2. Taormina, R., & Galelli, S. (2018). Deep-learning approach to the detection and localization of cyber-physical attacks on water distribution systems. *Journal of Water Resources Planning and Management*, 144(10).
3. Sikder, M. N. K., et al. (2023). Deep H2O: Cyber attacks detection in water distribution systems using deep learning. *Journal of Water Process Engineering*, 52, 103568.
4. Nemade, B., et al. (2026). Cyber-physical attack detection in water distribution systems using a hybrid ensemble of XGBoost, Isolation Forest, and LSTM autoencoder on the BATADAL dataset. *Journal of Intelligent Decision Making and Information Science*, 3(3s).
5. Habib, M. A., et al. (2026). Cybersecurity in water distribution networks: A systematic review of AI-based detection algorithms. *Water*, 18, 519.
6. Zhu, J., et al. (2026). Generative anomaly detection: A comprehensive review of modeling principles, advances, and future opportunities. *Artificial Intelligence Review*, 59, 193.
7. Vaswani, A., et al. (2017). Attention is all you need. arXiv:1706.03762.
8. Devlin, J., et al. (2018). BERT: Pre-training of deep bidirectional transformers for language understanding.
9. Dosovitskiy, A., et al. (2020). An image is worth 16x16 words: Transformers for image recognition at scale. arXiv:2010.11929.
10. Carion, N., et al. (2020). End-to-end object detection with transformers. arXiv:2005.12872.
11. Salici, G., et al. (2026). Segment-wise anomaly detection via compression tokens in industrial production lines. Ital-IA 2026. https://iris.unimore.it/retrieve/1222cb18-1ca8-4294-b112-8d34f859028b/salici_sew.pdf
12. Jaegle, A., et al. Perceiver IO: A general architecture for structured inputs and outputs.
13. Kaplan, J., et al. (2020). Scaling laws for neural language models. arXiv:2001.08361.

Papers added during the implementation review:

14. Pintilie, I., Manolache, A., & Brad, F. Time series anomaly detection using diffusion-based models. https://arxiv.org/pdf/2311.01452
15. Xu, J., et al. (2022). Anomaly Transformer: Time series anomaly detection with association discrepancy. ICLR 2022. https://arxiv.org/abs/2110.02642v3
16. Yu, Q., et al. (2024). An image is worth 32 tokens for reconstruction and generation. https://arxiv.org/pdf/2406.07550
17. Song, J., et al. (2023). MEMTO: Memory-guided Transformer for multivariate time series anomaly detection. NeurIPS 2023. https://proceedings.neurips.cc/paper_files/paper/2023/hash/b4c898eb1fb556b8d871fbe9ead92256-Abstract.html
18. Tuli, S., Casale, G., & Jennings, N. R. (2022). TranAD: Deep transformer networks for anomaly detection in multivariate time series data. VLDB 2022. https://arxiv.org/abs/2201.07284