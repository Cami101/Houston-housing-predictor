# House Price Prediction (Images + Tabular Features)

## Overview
Predict Houston house list prices using **images + tabular features**. You compare three neural setups—hybrid (CNN + MLP), image-only (CNN), and features-only (MLP)—and also train a classic Decision Tree baseline.

---

## Data & Preprocessing
- **Source files:** `home_data_train.csv`, `home_data_test.csv`, and `house_imgs/` (128×128 RGB).
- **Imputation:** `KNNImputer(n_neighbors=3)` for numeric columns.
- **Scaling:** `MinMaxScaler` on continuous features (`beds`, `baths`, `sqft`, `lot_size`, `year_built`).
- **Encoding:** One-hot encode `property_type` and `zipcode`; curated ~71 final features.
- **Images:** `ToTensor` then **mean-only** normalization (no std division).  
- **Augmentation (training only):** random rotation, brightness, contrast.

---

## Dataset & Loaders
- Custom `HouseImagesDataset` returns `{image, features (71-d), price}` for training; `{image, features}` for test.
- Split: **75% train / 25% val** via `random_split`.
- **Batch size:** 64; train shuffled, val/test non-shuffled.

---

## Models
1. **HybridHouseNN (image + features)**  
   - CNN stack on images → FC(2048→256).  
   - MLP on 71-d tabular features → (256).  
   - Concatenate → MLP head → **price**.
2. **HouseImageOnly (images only)**  
   - CNN stack → FC head → **price**.
3. **HouseFeatsOnly (tabular only)**  
   - MLP (71→256→256) → head → **price**.

> Note: A pretrained AlexNet/Res34-style hybrid variant is sketched but not included in the training loop.

---

## Training
- **Device:** CUDA if available.
- **Loss:** RMSE via `torchmetrics.MeanSquaredError(squared=False)`.
- **Optimizer:** Adam, `lr=0.02`, `weight_decay=1e-3`.
- **Epochs:** 30 for each model.
- Track epoch-wise **train** and **validation** losses.

---

## Results (Validation RMSE; ↓ is better)
- **Hybrid (images + features):** best ≈ **357k** by epoch 30.
- **Features-only:** best ≈ **365k** by epoch 30.
- **Image-only:** ≈ **860k–940k** (significantly worse).

**Takeaway:** Tabular features carry most signal. Images add a **small but real** improvement (~8–10k RMSE better than features-only at the end).

---

## Inference & Submission
- Build a test loader for `home_data_test.csv` (no target).
- Generate predictions (note: the last-trained model in the loop was **features-only**).  
- Save **`my_submission.csv`** with columns: `houseid, price`.
- Diagnostic distribution (example): `min ≈ 31,198`, `max ≈ 18,815,484`, with **61 values > $6M** → heavy-tail/outlier behavior.

---

## Baseline 2: Decision Tree Regressor
- Tabular-only baseline with selected features: `property_type`, `beds`, `baths`, `sqft`, `lot_size`, `year_built`, `latitude`, `longitude`.
- Preprocess: one-hot `property_type`, KNN impute.
- `DecisionTreeRegressor(max_depth=20)` → predictions on test; saved to `my_submission.csv`; plotted KDE of predicted prices.
- Provides a fast, interpretable non-neural comparison point.

---

## What Worked
- Solid performance from **feature engineering + MLP**.
- **Hybrid fusion** (images + features) outperformed features-only.
- RMSE made scale/fit intuitive.

## What Needs Tuning
- **Image-only** struggled—metadata is far more predictive for price.
- **LR 0.02** is quite high for Adam; loss still trending at 30 epochs.
- **ReLU on final output** can create dead zones and distort price scale; consider a **linear head**.
- Outlier predictions suggest need for robust target handling.

---

## Recommended Next Steps
1. **Predict log-price:** train on `log1p(price)` with a **linear output**; evaluate after `expm1`.
2. **Tune optimization:** Adam `lr≈1e-3`, scheduler (cosine/step), **early stopping** on val RMSE.
3. **Stronger CNN backbone:** ResNet18/34 (pretrained), lightweight fine-tuning, keep the tabular MLP as-is.
4. **Robustness:** Huber/SmoothL1 loss, gradient clipping; consider trimming extreme targets or use quantile loss.
5. **Richer tabular features:** interactions (e.g., `beds*sqft`), home age (`current_year - year_built`), neighborhood aggregates (zipcode medians), distances to POIs.

---

## TL;DR
- **Best so far:** Hybrid CNN+MLP → **val RMSE ≈ 357k**.  
- **Features-only** close behind (~365k); **images-only** underperforms.  
- Quick wins: **log-price + linear head + tuned LR/early stop + pretrained backbone**.

---

## Environment Notes
- PyTorch **2.1.0** (CUDA if available), TorchMetrics, scikit-learn, torchvision, seaborn, matplotlib, PIL, pandas, numpy.
