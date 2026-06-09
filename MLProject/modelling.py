import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              ConfusionMatrixDisplay, roc_curve,
                              precision_recall_curve, classification_report)
from mlflow.models.signature import infer_signature
import mlflow
import mlflow.sklearn

# ==============================================================================
# ARGUMENT PARSER
# ==============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--n_estimators',      type=int, default=100)
parser.add_argument('--max_depth',         type=int, default=10)
parser.add_argument('--min_samples_split', type=int, default=2)
parser.add_argument('--min_samples_leaf',  type=int, default=1)
args = parser.parse_args()

# ==============================================================================
# MLFLOW SETUP — pakai env var, bukan dagshub.init()
# ==============================================================================
DAGSHUB_USERNAME = "siilverred"
DAGSHUB_REPO     = "SMSML_Charlene-Silver"

mlflow.set_tracking_uri(
    f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow"
)

os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USERNAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.environ.get('DAGSHUB_USER_TOKEN', '')

EXPERIMENT_NAME = "IBM_HR_Attrition_CI"
mlflow.set_experiment(EXPERIMENT_NAME)

# ==============================================================================
# LOAD DATA
# ==============================================================================
DATA_DIR = "WA_Fn-UseC_-HR-Employee-Attrition_preprocessing"

X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
X_test  = pd.read_csv(f"{DATA_DIR}/X_test.csv")
y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").squeeze()
y_test  = pd.read_csv(f"{DATA_DIR}/y_test.csv").squeeze()

print(f"[+] Train: {X_train.shape}, Test: {X_test.shape}")

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['Retain', 'Attrition'])
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title('Confusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()

def plot_roc_curve(y_true, y_prob, auc_score, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC Curve (AUC = {auc_score:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve', fontsize=14)
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()

def plot_feature_importance(model, feature_names, save_path, top_n=20):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(top_n), top_importances[::-1], color='steelblue')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features[::-1])
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'Top {top_n} Feature Importances', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()

def save_classification_report(y_true, y_pred, save_path):
    report = classification_report(y_true, y_pred,
                                   target_names=['Retain', 'Attrition'])
    with open(save_path, 'w') as f:
        f.write(report)

# ==============================================================================
# TRAINING + MANUAL LOGGING
# ==============================================================================
os.makedirs("artifacts_tmp", exist_ok=True)

with mlflow.start_run(run_name="RandomForest_CI_Run") as run:
    print(f"[*] MLflow Run ID: {run.info.run_id}")

    # ── Parameters ─────────────────────────────────────────────────────────
    mlflow.log_param("n_estimators",      args.n_estimators)
    mlflow.log_param("max_depth",         args.max_depth)
    mlflow.log_param("min_samples_split", args.min_samples_split)
    mlflow.log_param("min_samples_leaf",  args.min_samples_leaf)
    mlflow.log_param("random_state",      42)
    mlflow.log_param("class_weight",      "balanced")

    # ── Train ───────────────────────────────────────────────────────────────
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth if args.max_depth > 0 else None,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    # ── Predict ─────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ── Metrics ─────────────────────────────────────────────────────────────
    acc         = accuracy_score(y_test, y_pred)
    prec        = precision_score(y_test, y_pred)
    rec         = recall_score(y_test, y_pred)
    f1          = f1_score(y_test, y_pred)
    auc         = roc_auc_score(y_test, y_prob)
    f1_macro    = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    prec_macro  = precision_score(y_test, y_pred, average='macro')
    rec_macro   = recall_score(y_test, y_pred, average='macro')

    mlflow.log_metric("accuracy",        acc)
    mlflow.log_metric("precision",       prec)
    mlflow.log_metric("recall",          rec)
    mlflow.log_metric("f1_score",        f1)
    mlflow.log_metric("roc_auc",         auc)
    mlflow.log_metric("f1_macro",        f1_macro)
    mlflow.log_metric("f1_weighted",     f1_weighted)
    mlflow.log_metric("precision_macro", prec_macro)
    mlflow.log_metric("recall_macro",    rec_macro)

    print(f"[+] Accuracy : {acc:.4f}")
    print(f"[+] F1 Score : {f1:.4f}")
    print(f"[+] ROC AUC  : {auc:.4f}")

    # ── Artifacts ───────────────────────────────────────────────────────────
    cm_path = "artifacts_tmp/confusion_matrix.png"
    plot_confusion_matrix(y_test, y_pred, cm_path)
    mlflow.log_artifact(cm_path, artifact_path="plots")

    roc_path = "artifacts_tmp/roc_curve.png"
    plot_roc_curve(y_test, y_prob, auc, roc_path)
    mlflow.log_artifact(roc_path, artifact_path="plots")

    fi_path = "artifacts_tmp/feature_importance.png"
    plot_feature_importance(model, list(X_train.columns), fi_path)
    mlflow.log_artifact(fi_path, artifact_path="plots")

    cr_path = "artifacts_tmp/classification_report.txt"
    save_classification_report(y_test, y_pred, cr_path)
    mlflow.log_artifact(cr_path, artifact_path="reports")

    # ── Log Model ───────────────────────────────────────────────────────────
    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        registered_model_name="IBM_Attrition_Model_CharleneSilver"
    )

    # ── Save model lokal untuk Docker ───────────────────────────────────────
    os.makedirs("model_output", exist_ok=True)
    import pickle
    with open("model_output/model.pkl", "wb") as f:
        pickle.dump(model, f)
    mlflow.log_artifact("model_output/model.pkl", artifact_path="model_pkl")

    print(f"[✓] Run selesai! Run ID: {run.info.run_id}")

import shutil
shutil.rmtree("artifacts_tmp")
print("[+] Done!")