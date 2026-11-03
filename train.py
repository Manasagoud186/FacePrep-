"""
ml/train.py — FacePrep ML Training Pipeline

Trains six classifiers on the interview feature dataset:
  1. K-Nearest Neighbours (KNN)
  2. Logistic Regression
  3. Support Vector Machine (SVM)
  4. Decision Tree
  5. Random Forest
  6. MLP Neural Network

Each model is wrapped in the preprocessing pipeline (imputation →
outlier clipping → scaling → OHE) so the pipeline can later be loaded
by the FastAPI backend as a single joblib artifact and used directly on
raw feature dicts.

Evaluation strategy (leakage-free):
  • Preprocessing fitted strictly on X_train → applied to X_test.
  • Stratified 80/20 hold-out split used for final test metrics.
  • Stratified 5-fold cross-validation on the full training set used for
    variance-aware CV metrics (separate from the hold-out test set).

Saved artefacts (ml/models/):
  knn_pipeline.joblib
  logistic_regression_pipeline.joblib
  svm_pipeline.joblib
  decision_tree_pipeline.joblib
  random_forest_pipeline.joblib
  mlp_pipeline.joblib
  best_pipeline.joblib          ← highest CV F1-macro model
  training_results.json         ← all raw metrics for evaluate.py

Usage:
  python ml/train.py            (from repo root)
  python train.py               (from ml/ directory)
"""

import os
import sys
import json
import time
from typing import Dict, Any, List

# Allow running from either repo root or ml/ directory
_THIS   = os.path.dirname(os.path.abspath(__file__))
_ROOT   = os.path.dirname(_THIS)
for _p in (_ROOT, _THIS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
import joblib

from sklearn.neighbors         import KNeighborsClassifier
from sklearn.linear_model      import LogisticRegression
from sklearn.svm               import SVC
from sklearn.tree              import DecisionTreeClassifier
from sklearn.ensemble          import RandomForestClassifier
from sklearn.neural_network    import MLPClassifier
from sklearn.pipeline          import Pipeline
from sklearn.model_selection   import StratifiedKFold, cross_validate
from sklearn.metrics           import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score,
    classification_report,
)
from sklearn.preprocessing     import label_binarize

from dataset       import get_dataset
from preprocessing import (
    build_pipeline, prepare_splits, encode_labels,
    CLASS_ORDER, FEATURE_COLS,
)

# -- Paths ---------------------------------------------------------------------

MODELS_DIR   = os.path.join(_THIS, "models")
RESULTS_JSON = os.path.join(MODELS_DIR, "training_results.json")

# -- Model definitions ---------------------------------------------------------

def _model_zoo() -> Dict[str, Any]:
    """
    Return dict of {model_key: unfitted_classifier}.
    Hyperparameters are sensible defaults — not tuned to the test set.
    """
    return {
        "knn": KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
            metric="euclidean",
        ),
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=42,
        ),
        "svm": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,   # needed for ROC-AUC
            random_state=42,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        ),
    }


# -- Pretty label for logging --------------------------------------------------

MODEL_LABELS = {
    "knn":                  "K-Nearest Neighbours",
    "logistic_regression":  "Logistic Regression",
    "svm":                  "Support Vector Machine",
    "decision_tree":        "Decision Tree",
    "random_forest":        "Random Forest",
    "mlp":                  "MLP Neural Network",
}


# -- Metric helpers ------------------------------------------------------------

def _compute_roc_auc(model_pipeline: Pipeline,
                     X_test: pd.DataFrame,
                     y_test: pd.Series) -> float:
    """
    OvR macro-averaged ROC-AUC.
    Reindexes predict_proba columns to match CLASS_ORDER so the
    binarised ground-truth and probability matrix align correctly.
    Returns 0.0 if the model has no predict_proba.
    """
    try:
        y_score_raw = model_pipeline.predict_proba(X_test)
        # Get the class order the pipeline's classifier learned
        clf_classes = list(model_pipeline.named_steps["clf"].classes_)
        # Build a reindex mapping: for each position in CLASS_ORDER
        # find its column index in y_score_raw
        col_idx = [clf_classes.index(c) for c in CLASS_ORDER if c in clf_classes]
        if len(col_idx) != len(CLASS_ORDER):
            return 0.0
        y_score = y_score_raw[:, col_idx]
        y_bin   = label_binarize(y_test, classes=CLASS_ORDER)
        return float(roc_auc_score(y_bin, y_score,
                                   multi_class="ovr", average="macro"))
    except Exception:
        return 0.0


def _compute_test_metrics(model_pipeline: Pipeline,
                           X_test: pd.DataFrame,
                           y_test: pd.Series) -> Dict[str, Any]:
    y_pred = model_pipeline.predict(X_test)

    acc  = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred,
                                 average="macro", zero_division=0))
    rec  = float(recall_score(y_test, y_pred,
                               average="macro", zero_division=0))
    f1   = float(f1_score(y_test, y_pred,
                           average="macro", zero_division=0))
    cm   = confusion_matrix(y_test, y_pred,
                             labels=CLASS_ORDER).tolist()
    roc  = _compute_roc_auc(model_pipeline, X_test, y_test)

    # Per-class metrics
    report_dict = classification_report(
        y_test, y_pred,
        labels=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        cls: {
            "precision": round(report_dict[cls]["precision"], 4),
            "recall":    round(report_dict[cls]["recall"],    4),
            "f1_score":  round(report_dict[cls]["f1-score"],  4),
            "support":   int(report_dict[cls]["support"]),
        }
        for cls in CLASS_ORDER
        if cls in report_dict
    }

    return {
        "accuracy":           round(acc,  4),
        "precision_macro":    round(prec, 4),
        "recall_macro":       round(rec,  4),
        "f1_macro":           round(f1,   4),
        "roc_auc_macro_ovr":  round(roc,  4),
        "confusion_matrix":   cm,
        "per_class":          per_class,
    }


def _compute_cv_metrics(model_key: str,
                         classifier: Any,
                         X_train: pd.DataFrame,
                         y_train: pd.Series,
                         preprocessor_pipeline: Pipeline,
                         cv_folds: int = 5) -> Dict[str, Any]:
    """
    Stratified K-Fold CV on the training set.
    The preprocessor is re-fitted inside each fold (handled by sklearn Pipeline).
    """
    cv_pipeline = Pipeline([
        ("preprocessor", preprocessor_pipeline.named_steps["preprocessor"]),
        ("clf",           classifier),
    ])

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    scoring = {
        "accuracy":  "accuracy",
        "precision": "precision_macro",
        "recall":    "recall_macro",
        "f1":        "f1_macro",
    }

    cv_results = cross_validate(
        cv_pipeline, X_train, y_train,
        cv=skf,
        scoring=scoring,
        return_train_score=False,
        n_jobs=1,
    )

    return {
        f"cv_{k}_mean": round(float(np.mean(cv_results[f"test_{k}"])), 4)
        for k in scoring.keys()
    } | {
        f"cv_{k}_std":  round(float(np.std(cv_results[f"test_{k}"])),  4)
        for k in scoring.keys()
    } | {
        "cv_folds": cv_folds,
        "cv_fold_scores": {
            k: [round(float(v), 4) for v in cv_results[f"test_{k}"]]
            for k in scoring.keys()
        },
    }


# -- Main pipeline -------------------------------------------------------------

def run_training() -> None:
    print("=" * 62)
    print("  FACEPREP  ·  ML CLASSIFICATION TRAINING PIPELINE")
    print("=" * 62)

    os.makedirs(MODELS_DIR, exist_ok=True)

    # -- 1. Load dataset -------------------------------------------------------
    print("\n[1/5]  Loading dataset ...")
    df = get_dataset(min_samples=300, save=True)

    n_total = len(df)
    n_real  = int((~df["is_synthetic"]).sum())
    n_synth = int(df["is_synthetic"].sum())
    label_counts = df["performance_label"].value_counts().to_dict()

    print(f"       Total rows   : {n_total}")
    print(f"       Real records : {n_real}")
    print(f"       Synthetic    : {n_synth}  (is_synthetic=True)")
    print(f"       Labels       : {label_counts}")

    # -- 2. Train/test split (stratified 80/20) --------------------------------
    print("\n[2/5]  Splitting data (stratified 80/20) ...")
    X_train, X_test, y_train, y_test = prepare_splits(df, test_size=0.20, random_state=42)

    # -- 3. Fit preprocessing on X_train ONLY ---------------------------------
    print("\n[3/5]  Fitting preprocessing pipeline on training set ...")
    base_pipe = build_pipeline()
    base_pipe.fit(X_train)
    print(f"       Feature matrix after transform: "
          f"{base_pipe.transform(X_train).shape[1]} columns")

    # -- 4. Train all models + evaluate ---------------------------------------
    print("\n[4/5]  Training & evaluating 6 classifiers ...\n")

    zoo      = _model_zoo()
    all_results: Dict[str, Dict] = {}
    best_key  = None
    best_f1   = -1.0

    for model_key, classifier in zoo.items():
        label = MODEL_LABELS[model_key]
        print(f"  -- {label} ------------------------------")

        # Full pipeline: preprocessor (refitted each CV fold) + classifier
        full_pipe = Pipeline([
            ("preprocessor", build_pipeline().named_steps["preprocessor"]),
            ("clf", classifier),
        ])

        t0 = time.time()
        full_pipe.fit(X_train, y_train)
        train_sec = time.time() - t0
        print(f"     Fit time      : {train_sec:.2f}s")

        # Hold-out test metrics
        test_metrics = _compute_test_metrics(full_pipe, X_test, y_test)
        print(f"     Accuracy      : {test_metrics['accuracy']:.4f}")
        print(f"     Precision     : {test_metrics['precision_macro']:.4f}  (macro)")
        print(f"     Recall        : {test_metrics['recall_macro']:.4f}  (macro)")
        print(f"     F1-macro      : {test_metrics['f1_macro']:.4f}")
        print(f"     ROC-AUC (OvR) : {test_metrics['roc_auc_macro_ovr']:.4f}")

        # K-Fold CV metrics (uses a fresh pipeline clone so no leakage)
        print(f"     Running 5-fold stratified CV ...")
        cv_metrics = _compute_cv_metrics(
            model_key, classifier, X_train, y_train, base_pipe, cv_folds=5
        )
        print(f"     CV F1-macro   : {cv_metrics['cv_f1_mean']:.4f} "
              f"(± {cv_metrics['cv_f1_std']:.4f})")

        # Save pipeline artefact
        artefact_path = os.path.join(MODELS_DIR, f"{model_key}_pipeline.joblib")
        joblib.dump(full_pipe, artefact_path)
        print(f"     Saved         : ml/models/{model_key}_pipeline.joblib")

        all_results[model_key] = {
            "label":       label,
            "fit_time_s":  round(train_sec, 2),
            "test":        test_metrics,
            "cv":          cv_metrics,
            "model_file":  f"{model_key}_pipeline.joblib",
        }

        if cv_metrics["cv_f1_mean"] > best_f1:
            best_f1  = cv_metrics["cv_f1_mean"]
            best_key = model_key

        print()

    # -- 5. Save best model + results JSON ------------------------------------
    print(f"[5/5]  Best model (CV F1-macro = {best_f1:.4f}): "
          f"{MODEL_LABELS[best_key]}")

    best_src  = os.path.join(MODELS_DIR, f"{best_key}_pipeline.joblib")
    best_dest = os.path.join(MODELS_DIR, "best_pipeline.joblib")
    best_pipe = joblib.load(best_src)
    joblib.dump(best_pipe, best_dest)

    # Also copy to top-level models/ for FastAPI backend
    top_models_dir = os.path.join(_ROOT, "models")
    os.makedirs(top_models_dir, exist_ok=True)
    joblib.dump(best_pipe, os.path.join(top_models_dir, "model_pipeline.joblib"))

    summary = {
        "dataset": {
            "total_rows": n_total,
            "real_rows":  n_real,
            "synth_rows": n_synth,
            "label_distribution": label_counts,
            "class_order": CLASS_ORDER,
            "features": FEATURE_COLS,
            "target": "performance_label",
            "split": "80/20 stratified",
            "cv": "5-fold stratified K-Fold",
        },
        "models": all_results,
        "best_model": {
            "key":          best_key,
            "label":        MODEL_LABELS[best_key],
            "cv_f1_macro":  round(best_f1, 4),
            "artefact":     "ml/models/best_pipeline.joblib",
            "backend_copy": "models/model_pipeline.joblib",
        },
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n       Results JSON  : ml/models/training_results.json")
    print(f"       Best pipeline : ml/models/best_pipeline.joblib")
    print(f"       Backend copy  : models/model_pipeline.joblib\n")
    print("=" * 62)
    print("  TRAINING COMPLETE — run  ml/evaluate.py  for full report")
    print("=" * 62)


# -- Entry point ---------------------------------------------------------------

if __name__ == "__main__":
    run_training()
