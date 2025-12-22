import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except:
    LGBMClassifier = None


class JobQualityEvaluator:
    """
    评估生成任务质量：
    - 分布吻合度 (KL 散度)
    - 随机森林真假分类能力
    - XGBoost 判别能力
    - LightGBM 判别能力
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.scaler = StandardScaler()

        self.rf = RandomForestClassifier(
            n_estimators=120,
            max_depth=8,
            n_jobs=-1
        )

        self.xgb = XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
        ) if XGBClassifier else None

        self.lgb = LGBMClassifier(
            n_estimators=150,
            max_depth=6
        ) if LGBMClassifier else None

    # ---------------------
    # 1) 计算分布 KL 散度
    # ---------------------
    def distribution_score(self, real_df: pd.DataFrame, fake_df: pd.DataFrame):
        score_list = []

        for col in real_df.columns:
            try:
                real_hist, _ = np.histogram(real_df[col], bins=20, density=True)
                fake_hist, _ = np.histogram(fake_df[col], bins=20, density=True)

                # 避免 0
                real_hist += 1e-9
                fake_hist += 1e-9

                kl = np.sum(real_hist * np.log(real_hist / fake_hist))

                score_list.append(np.exp(-kl))  # 越接近1表示越相似
            except:
                continue

        return np.mean(score_list) if score_list else 0.0

    # ---------------------
    # 2) 随机森林真假分类得分
    # ---------------------
    def rf_score(self, real_df: pd.DataFrame, fake_df: pd.DataFrame):

        real_df = real_df.dropna().copy()
        fake_df = fake_df.dropna().copy()

        real_df["label"] = 1
        fake_df["label"] = 0

        df = pd.concat([real_df, fake_df], axis=0)

        y = df["label"]
        X = df.drop("label", axis=1)

        X_scaled = self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        self.rf.fit(X_train, y_train)

        proba = self.rf.predict_proba(X_test)[:, 1]
        return 1 - log_loss(y_test, proba)

    # ---------------------
    # 3) XGBoost 评价
    # ---------------------
    def xgb_score(self, real_df, fake_df):
        if not self.xgb:
            return 0.0

        real_df = real_df.dropna().copy()
        fake_df = fake_df.dropna().copy()

        real_df["label"] = 1
        fake_df["label"] = 0

        df = pd.concat([real_df, fake_df], axis=0)

        y = df["label"]
        X = df.drop("label", axis=1)

        X_scaled = self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        self.xgb.fit(X_train, y_train)
        proba = self.xgb.predict_proba(X_test)[:, 1]

        return 1 - log_loss(y_test, proba)

    # ---------------------
    # 4) LightGBM 评价
    # ---------------------
    def lgb_score(self, real_df, fake_df):
        if not self.lgb:
            return 0.0

        real_df = real_df.dropna().copy()
        fake_df = fake_df.dropna().copy()

        real_df["label"] = 1
        fake_df["label"] = 0

        df = pd.concat([real_df, fake_df], axis=0)

        y = df["label"]
        X = df.drop("label", axis=1)

        X_scaled = self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        self.lgb.fit(X_train, y_train)
        proba = self.lgb.predict_proba(X_test)[:, 1]

        return 1 - log_loss(y_test, proba)

    # ---------------------
    # 5) 综合得分
    # ---------------------
    def overall_score(self, real_df, fake_df):
        dist = self.distribution_score(real_df, fake_df)
        rf = self.rf_score(real_df, fake_df)
        xgb = self.xgb_score(real_df, fake_df)
        lgb = self.lgb_score(real_df, fake_df)

        combined = (dist + rf + xgb + lgb) / 4.0

        return {
            "DistributionScore": dist,
            "RFScore": rf,
            "XGBScore": xgb,
            "LightGBMScore": lgb,
            "CombinedScore": combined
        }
