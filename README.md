# Food Delivery Time Prediction

This project predicts **how many minutes a food delivery will take**, using only information that
is known when the order is placed: the courier, the order, the distance and the conditions on the road.
It compares four machine learning models against a simple baseline and also gives a calibrated
uncertainty range (for example "25 ± 6 minutes") for each prediction.

## Dataset

- **Size:** 45,593 orders from 1,320 couriers in 22 Indian cities (11 Feb to 6 Apr 2022)
- **Target:** `Time_taken(min)`, delivery time in minutes (10 to 54, average 26.3)
- **Features used (12):** courier age, courier rating, distance (km), order hour, weather, road traffic,
  vehicle condition, type of order, type of vehicle, multiple deliveries, festival, city type

The dataset is included at `data/Food_delivery.csv`

## Libraries used

| Library | Used for |
|---|---|
| `pandas` | Loading and cleaning the data |
| `numpy` | Numerical work and the haversine distance formula |
| `scikit-learn` | Train/test split, scaling, Linear Regression, Decision Tree, Random Forest, hyperparameter tuning, metrics |
| `xgboost` | XGBoost model |
| `matplotlib`, `seaborn` | Plots (EDA in the notebook, result figure in `train.py`) |
| `joblib` | Saving the trained model and scaler (`.pkl`) |

Exact versions are in `requirements.txt`.

## Steps

1. **Cleaning:** fix text values (`conditions Sunny` → `Sunny`, `(min) 24` → 24, `"NaN"` → missing),
   clip 53 ratings above 5 down to 5.
2. **Feature engineering:** distance between restaurant and customer with the haversine formula, and
   the hour of the order. 431 orders with impossible distances (over 100 km, caused by wrong signs in
   their coordinates) are removed, leaving **45,162 orders**.
3. **Split:** random 80% train (36,129 orders) / 20% test (9,033 orders), `random_state=42`.
4. **Missing values:** filled with the median (numbers) or most common value (categories) of the
   **training set only**, so no information leaks from the test set.
5. **Encoding and scaling:** traffic as an ordered number (Low=0, Medium=1, High=2, Jam=3), one-hot
   encoding for the other categories (21 features in total), then `StandardScaler` fit on the training set.
6. **Models:** a mean baseline plus 4 models (below). Random Forest and XGBoost are tuned with
   `RandomizedSearchCV` (3-fold cross-validation on the training set only).
7. **Evaluation:** RMSE and R² on the test set, which is used only once at the end.

## Models used

A baseline plus **4 models**:

1. **Mean baseline**: always predicts the average delivery time (the minimum any model must beat)
2. **Linear Regression**
3. **Decision Tree**
4. **Random Forest** (default and tuned)
5. **XGBoost** (default and tuned)

## Results (test set, 9,033 orders)

| Model | R² | MAE (min) | RMSE (min) |
|---|---|---|---|
| Mean baseline | 0.000 | 7.64 | 9.47 |
| Linear Regression | 0.569 | 4.93 | 6.21 |
| Decision Tree | 0.666 | 4.18 | 5.46 |
| Random Forest (default) | 0.819 | 3.20 | 4.03 |
| XGBoost (default) | 0.822 | 3.18 | 4.00 |
| **Random Forest (tuned)** | **0.830** | **3.11** | **3.91** |
| XGBoost (tuned) | 0.829 | 3.12 | 3.91 |

- **R²** = share of the variation in delivery time the model explains (1.0 is perfect).
- **RMSE** = like MAE but punishes big mistakes more.

### R² of the 4 models

```
Linear Regression      0.569  ███████████▍
Decision Tree          0.666  █████████████▎
XGBoost (tuned)        0.829  ████████████████▌
Random Forest (tuned)  0.830  ████████████████▌   <- best
```

## Best model: tuned Random Forest

- **R² = 0.830**, average error **3.11 minutes**, which is less than half the error of the baseline (7.64).
- Best settings: `n_estimators=400`, `max_depth=20`, `max_features=0.5`, `min_samples_split=10`, `min_samples_leaf=1`.
- Tuned XGBoost is almost exactly as good (R² 0.829, difference of 0.002 minutes per order), so the two
  are effectively tied. Random Forest is kept as the final model.
- Linear Regression is much weaker (R² 0.57) because delivery time depends on combinations of
  factors (for example weather and traffic together), which tree models capture and a straight line cannot.
- The single Decision Tree fits the training data perfectly but does poorly on new data (overfitting).
  Random Forest fixes this by averaging 400 trees.

## Extra analysis

**Which features matter most?** Each group of features was removed in turn and the model retrained:

| Feature removed | Test RMSE increase (min) |
|---|---|
| Courier age and rating | +1.85 |
| Weather | +1.67 |
| Distance | +1.02 |
| Vehicle type and condition | +0.83 |
| Traffic | +0.27 |
| Multiple deliveries | +0.10 |
| Festival | +0.05 |
| City type, order hour, order type | about 0 |

**Prediction ranges (conformal prediction):** a range of **±6.2 minutes** around the prediction
contains the real delivery time for **89.9%** of test orders (target 90%). The range is too narrow in
heavy traffic (86 to 87%) and for orders with missing traffic information (64%). Using a separate
range for each traffic level (±5.2 min in low traffic up to ±7.0 min in jams) gives about 90% in
every group, at the same average width.

**Where the model makes mistakes:** only 1.2% of orders are off by more than 10 minutes. The biggest
errors are on orders with missing information (courier age, rating or traffic).

## Repository files

| File | What it is |
|---|---|
| `Food_Delivery_Time_Prediction.ipynb` | Notebook: data cleaning, EDA, all models, tuning |
| `train.py` | Script that reproduces every result above in one run |
| `requirements.txt` | Library versions |

## How to run

```bash
pip install -r requirements.txt
```

```bash
python train.py --data data/Food_delivery.csv
```

This takes about 4 minutes and writes `results/results.json` (all numbers),
`results/figures/pred_and_ablation.png` (result figure), and `results/rf_best.pkl` with
`results/standard_scaler.pkl` (trained model and scaler).

## AI use

Claude Code (Anthropic) was used to help
generate `train.py` and to help draft the report.
