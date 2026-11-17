"""Reproducible pipeline for the ES 678 paper.

Follows Food_Delivery_Time_Prediction.ipynb step by step (same cleaning, features,
encoding, random 80/20 split, scaler, models and tuning grids; tuned random forest
as the final model), with three additions:

  1. Missing values are filled AFTER the split, using training-set medians/modes
     only (the notebook filled them before the split, which leaks test data).
  2. A mean-prediction baseline.
  3. Graduate depth analysis: split-conformal prediction intervals (overall and
     per traffic level) and a feature-group ablation of the random forest.

Usage:
    python train.py --data data/Food_delivery.csv
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

SEED = 42
TARGET = "Time_taken(min)"
FEATURES = ["Delivery_person_Age", "Delivery_person_Ratings", "distance_km", "order_hour",
            "Weatherconditions", "Road_traffic_density", "Vehicle_condition", "Type_of_order",
            "Type_of_vehicle", "multiple_deliveries", "Festival", "City"]
NUM_IMPUTE = ["Delivery_person_Age", "Delivery_person_Ratings", "multiple_deliveries", "order_hour"]
CAT_IMPUTE = ["Weatherconditions", "Road_traffic_density", "Festival", "City"]
ONEHOT = ["Type_of_order", "Type_of_vehicle", "Festival", "City", "Weatherconditions"]
TRAFFIC = {"Low": 0, "Medium": 1, "High": 2, "Jam": 3}

# Notebook tuning grids
RF_GRID = {"n_estimators": [200, 300, 400, 500], "max_depth": [10, 15, 20, 25, None],
           "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4],
           "max_features": ["sqrt", "log2", 0.5]}
XGB_GRID = {"n_estimators": [300, 500, 800], "max_depth": [3, 4, 5, 6, 8],
            "learning_rate": [0.01, 0.03, 0.05, 0.1], "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0], "reg_alpha": [0, 0.1, 0.5, 1],
            "reg_lambda": [1, 1.5, 2]}

# Feature groups for the ablation (raw feature names)
GROUPS = {
    "courier (age, rating)": ["Delivery_person_Age", "Delivery_person_Ratings"],
    "distance": ["distance_km"],
    "order hour": ["order_hour"],
    "traffic": ["Road_traffic_density"],
    "weather": ["Weatherconditions"],
    "vehicle (type, condition)": ["Type_of_vehicle", "Vehicle_condition"],
    "multiple deliveries": ["multiple_deliveries"],
    "festival": ["Festival"],
    "city type": ["City"],
    "order type": ["Type_of_order"],
}


# ------------------------------------------------------------------ notebook steps
def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def load_and_clean(path):
    """Notebook sections 1, 2 and 4 (cleaning and feature engineering), minus imputation."""
    df = pd.read_csv(path)
    log = {"rows_raw": len(df)}

    # 'conditions Sunny' -> 'Sunny'
    df["Weatherconditions"] = df["Weatherconditions"].apply(
        lambda t: t.split(" ")[1] if pd.notnull(t) else t)
    # strip spaces in every text column; literal 'NaN' -> real missing value
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip()
        df[col] = df[col].replace("NaN", np.nan)
    # target '(min) 24' -> 24
    df[TARGET] = df[TARGET].apply(lambda t: t.split(" ")[1] if pd.notnull(t) else t).astype(int)
    for col in ["Delivery_person_Age", "Delivery_person_Ratings", "multiple_deliveries"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    log["rows_rating_gt5"] = int((df["Delivery_person_Ratings"] > 5).sum())
    df["Delivery_person_Ratings"] = df["Delivery_person_Ratings"].clip(upper=5)

    df["distance_km"] = haversine_km(df["Restaurant_latitude"], df["Restaurant_longitude"],
                                     df["Delivery_location_latitude"], df["Delivery_location_longitude"])
    log["rows_distance_ge_100km_dropped"] = int((df["distance_km"] >= 100).sum())
    df = df[df["distance_km"] < 100].reset_index(drop=True)

    df["order_hour"] = pd.to_datetime(df["Time_Orderd"], format="%H:%M:%S", errors="coerce").dt.hour
    log["rows_clean"] = len(df)
    log["missing_before_imputation"] = {c: int(df[c].isna().sum()) for c in FEATURES}
    return df, log


def impute(train, test):
    """Medians / modes learned from the training rows only, applied to both sets."""
    train, test = train.copy(), test.copy()
    fill = {}
    for c in NUM_IMPUTE:
        if c in train:
            fill[c] = float(train[c].median())
    for c in CAT_IMPUTE:
        if c in train:
            fill[c] = train[c].mode()[0]
    return train.fillna(fill), test.fillna(fill), fill


def encode(X, columns=None):
    """Notebook section 6: ordinal traffic + one-hot (drop_first) for the rest."""
    X = X.copy()
    if "Road_traffic_density" in X:
        X["Road_traffic_density"] = X["Road_traffic_density"].map(TRAFFIC)
    X = pd.get_dummies(X, columns=[c for c in ONEHOT if c in X], drop_first=True, dtype=int)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0)
    return X


# ----------------------------------------------------------------------- helpers
def metrics(y, p):
    return {"MAE": float(mean_absolute_error(y, p)),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "R2": float(r2_score(y, p))}


def bootstrap(y, preds, n=1000):
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    out = {}
    for name, p in preds.items():
        ae, se = np.abs(y - p), (y - p) ** 2
        mae, rmse = ae[idx].mean(1), np.sqrt(se[idx].mean(1))
        out[name] = {"MAE_CI": np.percentile(mae, [2.5, 97.5]).tolist(),
                     "RMSE_CI": np.percentile(rmse, [2.5, 97.5]).tolist()}
    return out, idx


def conformal_q(residuals, alpha):
    n = len(residuals)
    return float(np.quantile(residuals, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher"))


def prepare(train_raw, test_raw, feats):
    """Impute (train stats) -> encode -> scale (fit on train). Returns arrays."""
    tr, te, _ = impute(train_raw[feats], test_raw[feats])
    Xtr = encode(tr)
    Xte = encode(te, columns=Xtr.columns)
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte), Xtr.columns, sc


# -------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/Food_delivery.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--fast", action="store_true", help="small searches, for a quick test")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    R = {}

    df, R["cleaning"] = load_and_clean(args.data)
    X_raw, y = df[FEATURES], df[TARGET].to_numpy()

    # Notebook section 7: random 80/20 split
    tr_idx, te_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=SEED)
    Xtr_raw, Xte_raw = X_raw.iloc[tr_idx], X_raw.iloc[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    # Missing values filled from training rows only, then encode + scale (section 8)
    tr_imp, te_imp, R["imputation_values"] = impute(Xtr_raw, Xte_raw)
    X_train = encode(tr_imp)
    X_test = encode(te_imp, columns=X_train.columns)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    R["split"] = {"train_rows": len(tr_idx), "test_rows": len(te_idx),
                  "n_features_encoded": X_train.shape[1],
                  "train_target_mean": float(y_tr.mean()), "test_target_mean": float(y_te.mean())}
    print(json.dumps(R["split"]))

    # Models (notebook sections: Model 1-4 + tuning)
    n_rf, n_xgb = (3, 4) if args.fast else (25, 40)
    models = {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Linear regression": LinearRegression(),
        "Decision tree": DecisionTreeRegressor(random_state=SEED),
        "Random forest": RandomForestRegressor(random_state=SEED),
        "XGBoost": XGBRegressor(random_state=SEED),
    }
    for name, m in models.items():
        t = time.time()
        m.fit(X_train_scaled, y_tr)
        print(f"fit {name} ({time.time() - t:.0f}s)")

    t = time.time()
    rf_search = RandomizedSearchCV(RandomForestRegressor(random_state=SEED), RF_GRID, n_iter=n_rf,
                                   cv=3, scoring="r2", random_state=SEED, n_jobs=-1)
    rf_search.fit(X_train_scaled, y_tr)
    print(f"tuned RF ({time.time() - t:.0f}s): {rf_search.best_params_}")
    t = time.time()
    xgb_search = RandomizedSearchCV(XGBRegressor(random_state=SEED), XGB_GRID, n_iter=n_xgb,
                                    cv=3, scoring="r2", random_state=SEED, n_jobs=-1)
    xgb_search.fit(X_train_scaled, y_tr)
    print(f"tuned XGB ({time.time() - t:.0f}s): {xgb_search.best_params_}")
    models["Random forest (tuned)"] = rf_search.best_estimator_
    models["XGBoost (tuned)"] = xgb_search.best_estimator_
    R["tuning"] = {"Random forest": {"cv_r2": float(rf_search.best_score_), "best_params": rf_search.best_params_},
                   "XGBoost": {"cv_r2": float(xgb_search.best_score_), "best_params": xgb_search.best_params_}}

    preds, R["test"] = {}, {}
    for name, m in models.items():
        preds[name] = m.predict(X_test_scaled)
        R["test"][name] = metrics(y_te, preds[name]) | {"train": metrics(y_tr, m.predict(X_train_scaled))}
        print(f"{name:24s} test {R['test'][name]['MAE']:.3f} MAE  {R['test'][name]['RMSE']:.3f} RMSE  "
              f"{R['test'][name]['R2']:.4f} R2")
    cis, boot_idx = bootstrap(y_te, preds)
    for n in cis:
        R["test"][n].update(cis[n])

    best = "Random forest (tuned)"   # final model, as in the notebook
    R["final_model"] = best
    d = (np.abs(y_te - preds["XGBoost (tuned)"]) - np.abs(y_te - preds[best]))[boot_idx].mean(1)
    R["paired_mae_diff_xgb_minus_rf"] = {"mean": float(d.mean()),
                                         "CI": np.percentile(d, [2.5, 97.5]).tolist()}
    joblib.dump(rf_search.best_estimator_, out / "rf_best.pkl", compress=3)
    joblib.dump(scaler, out / "standard_scaler.pkl")

    # ---- error analysis (final model), using the raw labels of the test rows
    err = Xte_raw.copy()
    err["y"], err["pred"] = y_te, preds[best]
    err["abs_err"], err["resid"] = np.abs(err.y - err.pred), err.y - err.pred
    lab = lambda s: s.astype(object).where(s.notna(), "missing").astype(str)
    R["subgroups"] = {}
    for col in ["Road_traffic_density", "Festival", "City", "Weatherconditions", "Type_of_vehicle",
                "multiple_deliveries"]:
        g = err.groupby(lab(err[col])).agg(n=("abs_err", "size"), MAE=("abs_err", "mean"),
                                           bias=("resid", "mean"))
        R["subgroups"][col] = g.reset_index().rename(columns={"index": col}).to_dict(orient="records")
    q = pd.qcut(err.y, 5)
    R["error_by_true_quintile"] = err.groupby(q, observed=True).agg(
        n=("abs_err", "size"), MAE=("abs_err", "mean"), bias=("resid", "mean")).reset_index().astype(
        {"y": str}).to_dict(orient="records")
    R["share_abs_err_gt_10min"] = float((err.abs_err > 10).mean())
    worst = err.sort_values("abs_err", ascending=False).head(8)
    R["worst_errors"] = json.loads(worst.round(2).to_json(orient="records"))

    # ---- depth 1: split-conformal prediction intervals
    # Calibration rows are carved out of the TRAINING set (test set untouched).
    fit_i, cal_i = train_test_split(np.arange(len(tr_idx)), test_size=0.2, random_state=SEED)
    Xf, Xc, cols, _ = prepare(Xtr_raw.iloc[fit_i], Xtr_raw.iloc[cal_i], FEATURES)
    _, Xt, _, _ = prepare(Xtr_raw.iloc[fit_i], Xte_raw, FEATURES)
    cm = RandomForestRegressor(random_state=SEED, n_jobs=-1, **rf_search.best_params_).fit(Xf, y_tr[fit_i])
    cal_res = np.abs(y_tr[cal_i] - cm.predict(Xc))
    p_te = cm.predict(Xt)
    conf = {"n_fit": len(fit_i), "n_cal": len(cal_i)}
    for alpha in (0.2, 0.1, 0.05):
        qh = conformal_q(cal_res, alpha)
        conf[f"{round((1 - alpha) * 100)}%"] = {"halfwidth_min": qh,
                                               "coverage": float((np.abs(y_te - p_te) <= qh).mean())}
    q90 = conf["90%"]["halfwidth_min"]
    # Coverage of the single 90% interval by group, and per-traffic-level (Mondrian) intervals.
    # Traffic groups use the RAW label, so missing traffic is its own group.
    cal_tr = lab(Xtr_raw.iloc[cal_i]["Road_traffic_density"]).to_numpy()
    te_tr = lab(Xte_raw["Road_traffic_density"]).to_numpy()
    rows, mcov, mw = [], np.zeros(len(y_te), bool), np.zeros(len(y_te))
    for g in sorted(set(te_tr)):
        m = te_tr == g
        qg = conformal_q(cal_res[cal_tr == g], 0.1)
        mcov[m], mw[m] = np.abs(y_te[m] - p_te[m]) <= qg, qg
        rows.append({"traffic": g, "n_test": int(m.sum()), "n_cal": int((cal_tr == g).sum()),
                     "global_coverage": float((np.abs(y_te[m] - p_te[m]) <= q90).mean()),
                     "mondrian_halfwidth": qg, "mondrian_coverage": float(mcov[m].mean())})
    conf["by_traffic"] = rows
    conf["mondrian_overall"] = {"coverage": float(mcov.mean()), "mean_halfwidth": float(mw.mean())}
    for col in ["Festival", "City"]:
        lb = lab(Xte_raw[col]).to_numpy()
        conf[f"global_coverage_by_{col}"] = {g: {"n": int((lb == g).sum()),
                                                 "coverage": float((np.abs(y_te[lb == g] - p_te[lb == g]) <= q90).mean())}
                                             for g in sorted(set(lb))}
    R["conformal"] = conf

    # ---- depth 2: feature-group ablation (tuned RF settings, retrained without each group)
    full = R["test"][best]["RMSE"]
    abl = []
    for gname, cols_drop in GROUPS.items():
        feats = [f for f in FEATURES if f not in cols_drop]
        a_tr, a_te, _, _ = prepare(Xtr_raw, Xte_raw, feats)
        m = RandomForestRegressor(random_state=SEED, n_jobs=-1, **rf_search.best_params_).fit(a_tr, y_tr)
        r = metrics(y_te, m.predict(a_te))
        abl.append({"dropped": gname, **r, "delta_RMSE": r["RMSE"] - full})
        print(f"ablation -{gname:26s} RMSE {r['RMSE']:.3f} (+{r['RMSE'] - full:.3f})")
    R["ablation"] = sorted(abl, key=lambda a: -a["delta_RMSE"])

    # ---- figure
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.3))
    lim = [5, 60]
    axes[0].hexbin(y_te, preds[best], gridsize=40, extent=lim + lim, mincnt=1, cmap="Blues", bins="log")
    axes[0].plot(lim, lim, color="#c0392b", lw=1)
    axes[0].set_xlabel("Actual time (min)")
    axes[0].set_ylabel("Predicted time (min)")
    axes[0].set_title("Tuned random forest, test set", fontsize=8)
    a = R["ablation"]
    axes[1].barh([x["dropped"] for x in a][::-1], [x["delta_RMSE"] for x in a][::-1], color="#2a6fb0")
    axes[1].axvline(0, color="#333", lw=0.6)
    axes[1].set_xlabel("Test RMSE increase when dropped (min)")
    axes[1].set_title("Feature-group ablation", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figures" / "pred_and_ablation.pdf")
    fig.savefig(out / "figures" / "pred_and_ablation.png", dpi=200)

    R["runtime_sec"] = round(time.time() - t0, 1)
    (out / "results.json").write_text(json.dumps(R, indent=2, default=str))
    print(f"done in {R['runtime_sec']}s -> {out / 'results.json'}")


if __name__ == "__main__":
    main()
