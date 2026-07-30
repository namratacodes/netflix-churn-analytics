import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
import shap


CHURN_FEATURES = [
    "obs_avg_sessions_per_week", "avg_completion_rate", "genre_diversity",
    "tenure_days", "obs_recent_30d_sessions", "obs_prior_30d_sessions",
    "obs_session_trend_ratio",
]


def split_churn_data(df, feature_cols=CHURN_FEATURES, label_col="churned_next_30d",
                      test_size=0.2, random_state=42):
    
    X = df[feature_cols]
    y = df[label_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def train_churn_model(X_train, y_train, n_estimators=200, max_depth=8, random_state=42):
    
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=random_state,
    )
    rf.fit(X_train, y_train)
    return rf


def evaluate_churn_model(model, X_test, y_test):
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_pred_proba)

    return {
        "classification_report": report,
        "roc_auc": auc,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
    }


def explain_with_shap(model, X_test):
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap_values_churn = shap_values[:, :, 1]
    return explainer, shap_values_churn