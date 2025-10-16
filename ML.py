# -*- coding: utf-8 -*-

# ===================== CONFIG =====================
DATA_LABEL   = "dist30-svm"
CSV_PATH     = "phase_descriptors_dist30.csv"
MODEL_TYPE   = "svm"   # "svm" | "knn" | "rf" | "gb"

RANDOM_STATE   = 42
N_TRIALS       = 95           # Optuna trials
N_BOOTSTRAPS   = 1000         # resamples for 95% CI on hold-out
N_FOLDS_IMPORT = 5            # folds for aggregated permutation importance

# ===================== IMPORTS =====================
import hashlib, json, platform
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import optuna
import matplotlib
import matplotlib.pyplot as plt
import sklearn

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef, cohen_kappa_score,
    confusion_matrix, classification_report
)
from sklearn.inspection import permutation_importance

# ===================== OUTPUT FOLDER & FIG STYLE =====================
RESULTS_DIR = Path(DATA_LABEL); RESULTS_DIR.mkdir(parents=True, exist_ok=True)
matplotlib.rcParams.update({"font.family": "Times New Roman", "figure.dpi": 300, "savefig.dpi": 300})

# ===================== FEATURES / TARGET =====================
DESCRIPTORS = [
    "Hfus_avgs_w","Hfus_std_devs_w","Sfus_avgs_w","Sfus_std_devs_w","Sid",
    "Electronegativity_avgs_w","Electronegativity_std_devs_w",
    "VEC_avgs_w","VEC_std_devs_w",
    "AtomicRadius_avgs_w","AtomicRadius_std_devs_w",
    "AtomicWeight_avgs_w","AtomicWeight_std_devs_w",
    "AtomicSizeMismatchs",
    "MeltT_avgs_w","MeltT_std_devs_w",
    "BoilingT_avgs_w","BoilingT_std_devs_w",
    "Hmix_l","exCp_l"
]
TARGET = "phase_score"

# ===================== DATA LOADING & SPLIT =====================
df = pd.read_csv(CSV_PATH)
X = df[DESCRIPTORS].copy()
y = df[TARGET].copy()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

# ===================== MODEL / PIPELINE FACTORY =====================
def make_pipeline(model_type: str, **params):
    """
    Builds a Pipeline per model type.
    - Scaling is applied for SVM/KNN; skipped for tree models.
    """
    if model_type == "svm":
        clf = SVC(
            C=params.get("C", 1.0),
            kernel=params.get("kernel", "rbf"),
            gamma=params.get("gamma", "scale"),
            degree=params.get("degree", 3),
            class_weight=params.get("class_weight", None),
            probability=False,
            cache_size=1000,
            random_state=RANDOM_STATE
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    if model_type == "knn":
        clf = KNeighborsClassifier(
            n_neighbors=params.get("n_neighbors", 5),
            weights=params.get("weights", "uniform"),
            metric=params.get("metric", "euclidean"),
            p=params.get("p", 2),
            leaf_size=params.get("leaf_size", 30),
            n_jobs=-1
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    if model_type == "rf":
        clf = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 500),
            max_depth=params.get("max_depth", None),
            min_samples_split=params.get("min_samples_split", 2),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            max_features=params.get("max_features", "sqrt"),
            bootstrap=True,
            class_weight=params.get("class_weight", None),
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
        return Pipeline([("clf", clf)])  # no scaling for trees

    if model_type == "gb":
        clf = GradientBoostingClassifier(
            n_estimators=params.get("n_estimators", 300),
            learning_rate=params.get("learning_rate", 0.1),
            max_depth=params.get("max_depth", 3),
            subsample=params.get("subsample", 1.0),
            max_features=params.get("max_features", None),
            random_state=RANDOM_STATE
        )
        return Pipeline([("clf", clf)])  # no scaling for trees

    raise ValueError(f"Unsupported model_type: {model_type}")

# ===================== OPTUNA OBJECTIVE (INNER CV) =====================
skf_inner = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)

def objective(trial):
    """
    Defines a typical search space per model and returns CV mean F1-weighted.
    """
    if MODEL_TYPE == "svm":
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly"])
        params = {
            "C":     trial.suggest_float("C", 1e-3, 1e3, log=True),
            "kernel": kernel,
            "gamma":  trial.suggest_categorical("gamma", ["scale", "auto"]) if kernel in ["rbf","poly"] else "scale",
            "degree": trial.suggest_int("degree", 2, 5) if kernel == "poly" else 3,
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }

    elif MODEL_TYPE == "knn":
        metric = trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"])
        params = {
            "n_neighbors": trial.suggest_int("n_neighbors", 1, 60),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "metric": metric,
            "p": trial.suggest_int("p", 1, 5) if metric == "minkowski" else (1 if metric=="manhattan" else 2),
            "leaf_size": trial.suggest_int("leaf_size", 15, 60),
        }

    elif MODEL_TYPE == "rf":
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 1000),
            "max_depth":         trial.suggest_int("max_depth", 3, 40),  # use None by allowing large depth
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "class_weight":      trial.suggest_categorical("class_weight", [None, "balanced"]),
        }

    elif MODEL_TYPE == "gb":
        params = {
            "n_estimators":  trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth":     trial.suggest_int("max_depth", 2, 5),
            "subsample":     trial.suggest_float("subsample", 0.6, 1.0),
            "max_features":  trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
        }

    else:
        raise ValueError(f"Unsupported MODEL_TYPE: {MODEL_TYPE}")

    pipe = make_pipeline(MODEL_TYPE, **params)
    scores = cross_val_score(pipe, X_train, y_train, cv=skf_inner, scoring="f1_weighted", n_jobs=-1)
    return scores.mean()

# ===================== RUN HPO (SEEDED + PRUNER) =====================
sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
pruner  = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=5, reduction_factor=3)
study   = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
study.optimize(objective, n_trials=N_TRIALS)

best_params = study.best_params.copy()
if MODEL_TYPE == "svm":
    if best_params.get("kernel") != "poly": best_params["degree"] = 3
    if best_params.get("kernel") not in ["rbf","poly"]: best_params["gamma"] = "scale"

# ===================== REFIT ON FULL TRAIN =====================
clf = make_pipeline(MODEL_TYPE, **best_params)
clf.fit(X_train, y_train)

# ===================== HOLD-OUT EVALUATION (95% CI VIA BOOTSTRAP) =====================
y_pred = clf.predict(X_test)
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
    "f1_macro": f1_score(y_test, y_pred, average="macro"),
    "mcc": matthews_corrcoef(y_test, y_pred),
    "kappa": cohen_kappa_score(y_test, y_pred),
}

rng = np.random.RandomState(RANDOM_STATE)

def stratified_bootstrap_indices(y, B):
    y = np.asarray(y); classes, inv = np.unique(y, return_inverse=True)
    idx_by_class = [np.where(inv==c)[0] for c in range(len(classes))]
    for _ in range(B):
        yield np.concatenate([rng.choice(idx, size=len(idx), replace=True) for idx in idx_by_class])

def bootstrap_ci(metric_fn, y_true, y_hat, B=1000, alpha=0.05):
    vals = [metric_fn(y_true[idx], y_hat[idx]) for idx in stratified_bootstrap_indices(y_true, B)]
    lo, hi = np.percentile(vals, [100*alpha/2, 100*(1-alpha/2)])
    return float(np.mean(vals)), (float(lo), float(hi))

_, acc_ci = bootstrap_ci(lambda yt, yp: accuracy_score(yt, yp), y_test.to_numpy(), y_pred, B=N_BOOTSTRAPS)
_, f1w_ci = bootstrap_ci(lambda yt, yp: f1_score(yt, yp, average="weighted"), y_test.to_numpy(), y_pred, B=N_BOOTSTRAPS)

# ===================== CONFUSION MATRIX (COUNTS + NORMALIZED) =====================
labels = np.unique(np.concatenate([y_test, y_pred]))
cm_counts = confusion_matrix(y_test, y_pred, labels=labels)
cm_norm   = confusion_matrix(y_test, y_pred, labels=labels, normalize="true")

cm_counts_df = pd.DataFrame(cm_counts, index=[f"true_{c}" for c in labels], columns=[f"pred_{c}" for c in labels])
cm_norm_df   = pd.DataFrame(cm_norm,   index=[f"true_{c}" for c in labels], columns=[f"pred_{c}" for c in labels])

with pd.ExcelWriter(RESULTS_DIR / f"confusion_matrix_{DATA_LABEL}_{MODEL_TYPE}.xlsx", engine="openpyxl") as w:
    cm_counts_df.to_excel(w, sheet_name="counts")
    cm_norm_df.to_excel(w,   sheet_name="normalized")

fig, ax = plt.subplots(figsize=(6,6))
im = ax.imshow(cm_norm, interpolation="nearest"); ax.set_title("Confusion Matrix (normalized)")
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
cbar = plt.colorbar(im, ax=ax); cbar.ax.set_ylabel("Proportion", rotation=90, va="center")
plt.tight_layout()
fig.savefig(RESULTS_DIR / f"confusion_matrix_normalized_{DATA_LABEL}_{MODEL_TYPE}.tif", format="tif")
fig.savefig(RESULTS_DIR / f"confusion_matrix_normalized_{DATA_LABEL}_{MODEL_TYPE}.pdf")
plt.close(fig)

# ===================== FEATURE IMPORTANCE (PERMUTATION, AGGREGATED) =====================
skf_eval = StratifiedKFold(n_splits=N_FOLDS_IMPORT, shuffle=True, random_state=RANDOM_STATE)
imps = []
for tr_idx, va_idx in skf_eval.split(X_train, y_train):
    Xtr, Xva = X_train.iloc[tr_idx], X_train.iloc[va_idx]
    ytr, yva = y_train.iloc[tr_idx], y_train.iloc[va_idx]
    model = make_pipeline(MODEL_TYPE, **best_params).fit(Xtr, ytr)
    pi = permutation_importance(model, Xva, yva, scoring="f1_weighted",
                                n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
    imps.append(pi.importances_mean)

imp_mean = np.mean(np.vstack(imps), axis=0)
imp_std  = np.std(np.vstack(imps), axis=0)

fi_df = (pd.DataFrame({"feature": DESCRIPTORS, "importance_mean": imp_mean, "importance_std": imp_std})
         .sort_values("importance_mean", ascending=False).reset_index(drop=True))

fi_df.to_excel(RESULTS_DIR / f"feature_importance_permutation_{DATA_LABEL}_{MODEL_TYPE}.xlsx", index=False)

fig, ax = plt.subplots(figsize=(8,6))
ax.barh(fi_df["feature"], fi_df["importance_mean"], xerr=fi_df["importance_std"])
ax.invert_yaxis(); ax.set_xlabel("Permutation Importance (Δ F1-weighted)")
ax.set_title(f"Feature Importance (aggregated over {N_FOLDS_IMPORT} folds)")
plt.tight_layout()
fig.savefig(RESULTS_DIR / f"feature_importance_permutation_{DATA_LABEL}_{MODEL_TYPE}.tif", format="tif")
fig.savefig(RESULTS_DIR / f"feature_importance_permutation_{DATA_LABEL}_{MODEL_TYPE}.pdf")
plt.close(fig)

# ===================== ARTIFACTS & LOGS =====================
joblib.dump(clf, RESULTS_DIR / f"final_model_{DATA_LABEL}_{MODEL_TYPE}.joblib")
pd.Series(best_params, name="value").to_csv(RESULTS_DIR / f"best_params_{DATA_LABEL}_{MODEL_TYPE}.csv")
study.trials_dataframe().to_csv(RESULTS_DIR / f"optuna_trials_{DATA_LABEL}_{MODEL_TYPE}.csv", index=False)

summary = [
    "=== Best hyperparameters (Optuna) ===", str(best_params), "",
    "=== Test metrics ===",
    f"Accuracy: {metrics['accuracy']:.6f}  (95% CI {acc_ci[0]:.6f}–{acc_ci[1]:.6f})",
    f"F1-weighted: {metrics['f1_weighted']:.6f}  (95% CI {f1w_ci[0]:.6f}–{f1w_ci[1]:.6f})",
    f"F1-macro: {metrics['f1_macro']:.6f}",
    f"MCC: {metrics['mcc']:.6f}",
    f"Cohen's kappa: {metrics['kappa']:.6f}",
    f"n_trials: {len(study.trials)}",
    f"model_type: {MODEL_TYPE}"
]
(RESULTS_DIR / f"summary_{DATA_LABEL}_{MODEL_TYPE}.txt").write_text("\n".join(summary), encoding="utf-8")
(RESULTS_DIR / f"classification_report_{DATA_LABEL}_{MODEL_TYPE}.txt").write_text(
    classification_report(y_test, y_pred, digits=4), encoding="utf-8"
)

env_meta = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "sklearn": sklearn.__version__, "optuna": optuna.__version__}
(RESULTS_DIR / f"env_versions_{DATA_LABEL}_{MODEL_TYPE}.json").write_text(json.dumps(env_meta, indent=2), encoding="utf-8")

data_hash = hashlib.sha256(pd.read_csv(CSV_PATH).to_csv(index=False).encode()).hexdigest()
(RESULTS_DIR / f"data_hash_{DATA_LABEL}_{MODEL_TYPE}.txt").write_text(data_hash, encoding="utf-8")

print(f"\n✔ Done. Results saved under: {RESULTS_DIR.resolve()} | Model: {MODEL_TYPE}")
