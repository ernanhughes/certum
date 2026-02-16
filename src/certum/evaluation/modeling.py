"""
Model training utilities for Certum evaluation.
"""

from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def run_model(
    df,
    features: List[str],
    test_size: float = 0.3,
    random_state: int = 42,
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
    """
    Train/test split logistic regression model.

    Returns:
        auc,
        coefficient_dict,
        y_test,
        predicted_probabilities
    """

    X = df[features].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=3000)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)

    coefs = dict(zip(features, model.coef_[0]))

    return auc, coefs, y_test, probs


def run_model_cv(
    df,
    features: List[str],
    folds: int = 5,
    random_state: int = 42,
) -> Tuple[float, float]:
    """
    Stratified k-fold cross-validation AUC.
    Returns mean and std.
    """

    X = df[features].values
    y = df["label"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    skf = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=random_state,
    )

    aucs = []

    for train_idx, test_idx in skf.split(X, y):
        model = LogisticRegression(max_iter=3000)
        model.fit(X[train_idx], y[train_idx])

        probs = model.predict_proba(X[test_idx])[:, 1]
        auc = roc_auc_score(y[test_idx], probs)
        aucs.append(auc)

    return float(np.mean(aucs)), float(np.std(aucs))




def run_xgb_model(df, features, seed=42):

    X = df[features].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        eval_metric="logloss",
    )

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)

    importances = dict(zip(features, model.feature_importances_))

    return auc, importances, y_test, probs


def run_xgb_model_cv(df, features, seed=42, n_splits=5):

    X = df[features].values
    y = df["label"].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    aucs = []

    for train_idx, test_idx in skf.split(X, y):
        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed,
            eval_metric="logloss",
        )

        model.fit(X[train_idx], y[train_idx])
        probs = model.predict_proba(X[test_idx])[:, 1]
        auc = roc_auc_score(y[test_idx], probs)
        aucs.append(auc)

    return np.mean(aucs), np.std(aucs)
