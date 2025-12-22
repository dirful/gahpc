#!/usr/bin/env python3
"""
anomaly.py - 异常检测
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

def detect_anomalies_iforest(
        features: np.ndarray,
        contamination: float = 0.03,
        n_jobs: int = -1
) -> np.ndarray:
    """使用孤立森林检测异常"""
    clf = IsolationForest(contamination=contamination,
                          n_estimators=200,
                          random_state=42,
                          n_jobs=n_jobs)
    pred = clf.fit_predict(features)
    return (pred == -1).astype(int)

def detect_anomalies_ocsvm(
        features: np.ndarray,
        nu: float = 0.01,
        kernel: str = 'rbf'
) -> np.ndarray:
    """使用One-Class SVM检测异常"""
    clf = OneClassSVM(nu=nu, kernel=kernel)
    pred = clf.fit_predict(features)
    return (pred == -1).astype(int)

def add_anomaly_columns(
        emb_df: pd.DataFrame,
        embeddings: np.ndarray
) -> pd.DataFrame:
    """添加异常检测列到嵌入数据框"""
    emb_df = emb_df.copy()

    # 孤立森林异常检测
    iso_anom = detect_anomalies_iforest(embeddings)
    emb_df['anomaly_iforest'] = iso_anom

    # One-Class SVM异常检测
    try:
        ocsvm_anom = detect_anomalies_ocsvm(embeddings)
        emb_df['anomaly_ocsvm'] = ocsvm_anom
    except Exception:
        emb_df['anomaly_ocsvm'] = np.zeros(len(embeddings), dtype=int)

    return emb_df