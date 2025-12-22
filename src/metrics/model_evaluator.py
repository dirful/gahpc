# src/metrics/model_evaluator.py
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from typing import Optional, Dict, Any

# optional boosters
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

# optional scientific helpers
try:
    from scipy.stats import wasserstein_distance as _scipy_wasserstein
except Exception:
    _scipy_wasserstein = None

try:
    from scipy.linalg import sqrtm as _scipy_sqrtm
except Exception:
    _scipy_sqrtm = None

logger = logging.getLogger(__name__)


def _safe_numeric_df(real: pd.DataFrame, fake: pd.DataFrame):
    """返回仅包含两表交集的数值列的 DataFrame 副本（并去掉 NA）"""
    r = real.copy() if isinstance(real, pd.DataFrame) else pd.DataFrame()
    f = fake.copy() if isinstance(fake, pd.DataFrame) else pd.DataFrame()

    # 只保留数值列
    r_num = r.select_dtypes(include=[np.number])
    f_num = f.select_dtypes(include=[np.number])

    # 对齐列：取交集（以真实数据列为准）
    common_cols = [c for c in r_num.columns if c in f_num.columns]
    if not common_cols:
        return pd.DataFrame(), pd.DataFrame(), []

    r_sel = r_num[common_cols].dropna()
    f_sel = f_num[common_cols].dropna()

    return r_sel, f_sel, common_cols


def _shared_hist_kl(p_samples: np.ndarray, q_samples: np.ndarray, bins=20, eps=1e-9):
    """
    使用共享 bins 计算 KL(p||q)，返回 kl 值。
    p_samples, q_samples: 1D arrays
    """
    p_samples = np.asarray(p_samples).ravel()
    q_samples = np.asarray(q_samples).ravel()

    if p_samples.size == 0 or q_samples.size == 0:
        return np.nan

    mn = float(np.min([p_samples.min(), q_samples.min()]))
    mx = float(np.max([p_samples.max(), q_samples.max()]))
    if mn == mx:
        # 常数分布，KL 视为 0
        return 0.0

    bins_edges = np.linspace(mn, mx, bins + 1)
    p_hist, _ = np.histogram(p_samples, bins=bins_edges, density=True)
    q_hist, _ = np.histogram(q_samples, bins=bins_edges, density=True)

    p_hist = p_hist + eps
    q_hist = q_hist + eps

    kl = np.sum(p_hist * np.log(p_hist / q_hist))
    return float(kl)


def _rbf_kernel(X, Y=None, gamma=None):
    """
    RBF kernel matrix between X and Y.
    X: (n, d), Y: (m, d) or None -> returns (n,m) or (n,n) if Y is None
    gamma: kernel scale. If None, set to 1 / (d * median_sq_dist)
    """
    X = np.asarray(X, dtype=np.float64)
    if Y is None:
        Y = X
    else:
        Y = np.asarray(Y, dtype=np.float64)

    if X.size == 0 or Y.size == 0:
        return np.zeros((X.shape[0], Y.shape[0]))

    # pairwise squared euclidean distances
    XX = np.sum(X ** 2, axis=1)[:, None]
    YY = np.sum(Y ** 2, axis=1)[None, :]
    D2 = XX + YY - 2.0 * (X @ Y.T)
    D2 = np.maximum(D2, 0.0)

    if gamma is None:
        # median heuristic
        vals = D2.flatten()
        vals = vals[vals > 0]
        if len(vals) == 0:
            gamma = 1.0
        else:
            median = np.median(vals)
            if median <= 0:
                gamma = 1.0 / (X.shape[1] + 1e-8)
            else:
                gamma = 1.0 / (2.0 * median)
    K = np.exp(-gamma * D2)
    return K


def _mmd_rbf(X, Y, gamma=None):
    """
    Compute squared MMD with RBF kernel: E[k(X,X)] + E[k(Y,Y)] - 2 E[k(X,Y)]
    Returns scalar (>=0)
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.size == 0 or Y.size == 0:
        return np.nan

    Kxx = _rbf_kernel(X, X, gamma)
    Kyy = _rbf_kernel(Y, Y, gamma)
    Kxy = _rbf_kernel(X, Y, gamma)

    # unbiased estimates (diagonal removed)
    n = Kxx.shape[0]
    m = Kyy.shape[0]
    if n > 1:
        term_x = (np.sum(Kxx) - np.trace(Kxx)) / (n * (n - 1))
    else:
        term_x = np.sum(Kxx) / (n * n)
    if m > 1:
        term_y = (np.sum(Kyy) - np.trace(Kyy)) / (m * (m - 1))
    else:
        term_y = np.sum(Kyy) / (m * m)

    term_xy = np.sum(Kxy) / (n * m)
    mmd2 = term_x + term_y - 2.0 * term_xy
    return float(max(mmd2, 0.0))


def _wasserstein_1d(u: np.ndarray, v: np.ndarray):
    """
    1D wasserstein (earth mover) distance.
    If scipy available, uses scipy.stats.wasserstein_distance.
    Otherwise use a simple empirical CDF integration approach.
    """
    u = np.asarray(u, dtype=np.float64).ravel()
    v = np.asarray(v, dtype=np.float64).ravel()
    if u.size == 0 or v.size == 0:
        return np.nan

    if _scipy_wasserstein is not None:
        try:
            return float(_scipy_wasserstein(u, v))
        except Exception:
            pass

    # fallback: empirical integration on sorted samples
    u_sorted = np.sort(u)
    v_sorted = np.sort(v)
    # build uniform cdf positions
    all_vals = np.concatenate([u_sorted, v_sorted])
    # compute unique sorted positions
    xs = np.unique(all_vals)
    # empirical CDFs
    cu = np.searchsorted(u_sorted, xs, side='right') / float(max(1, u_sorted.size))
    cv = np.searchsorted(v_sorted, xs, side='right') / float(max(1, v_sorted.size))
    # integrate absolute difference via trapezoid on xs
    diffs = np.diff(xs)
    if diffs.size == 0:
        # degenerate: all equal
        return float(np.abs(u_sorted.mean() - v_sorted.mean()))
    mid_diff = 0.5 * (np.abs(cu[:-1] - cv[:-1]) + np.abs(cu[1:] - cv[1:]))
    return float(np.sum(mid_diff * diffs))


def _frechet_distance(X: np.ndarray, Y: np.ndarray):
    """
    Frechet distance between two multivariate Gaussians estimated from X and Y:
    d^2 = ||mu_x - mu_y||^2 + Tr(Cov_x + Cov_y - 2*sqrt(Cov_x*Cov_y))
    Returns scalar distance (not squared).
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.size == 0 or Y.size == 0:
        return np.nan

    mu_x = np.mean(X, axis=0)
    mu_y = np.mean(Y, axis=0)

    cov_x = np.cov(X, rowvar=False)
    cov_y = np.cov(Y, rowvar=False)

    # ensure 2D
    if cov_x.ndim == 0:
        cov_x = np.atleast_2d(cov_x)
    if cov_y.ndim == 0:
        cov_y = np.atleast_2d(cov_y)

    try:
        if _scipy_sqrtm is not None:
            covmean = _scipy_sqrtm(cov_x.dot(cov_y))
            # handle numerical error (small imaginary component)
            if np.iscomplexobj(covmean):
                covmean = covmean.real
        else:
            # fallback: approximate sqrt by eigen decomposition (may be unstable)
            w, v = np.linalg.eigh(cov_x.dot(cov_y))
            w = np.where(w < 0, 0.0, w)
            covmean = (v * np.sqrt(w)) @ v.T
    except Exception:
        # fallback: use trace approximation
        covmean = np.zeros_like(cov_x)

    diff = mu_x - mu_y
    tr = np.trace(cov_x + cov_y - 2.0 * covmean)
    tr = float(np.real_if_close(tr))
    dist2 = np.dot(diff, diff) + tr
    dist2 = max(dist2, 0.0)
    return float(np.sqrt(dist2))


class JobQualityEvaluator:
    """
    Enhanced evaluator for generated job quality.

    Methods:
      - distribution_score: average per-feature exp(-KL) using shared bins
      - mmd_score: squared MMD (RBF) between real and fake (lower is better)
      - wasserstein_score: average per-feature 1D Wasserstein (lower better)
      - frechet_score: Frechet distance between multivariate Gaussians (lower better)
      - classifier scores: 1-AUC for RF/XGB/LGB (closer to 1 = indistinguishable)
      - overall_score: markdown table + structured return

    Configurable via cfg:
      - evaluator_weights: dict with keys 'distribution','mmd','wasserstein','frechet','rf','xgb','lgb'
      - n_bins: bins for KL histogram
      - mmd_gamma: optional gamma for RBF kernel
      - random_state: for train_test_split / classifiers
    """

    def __init__(self, cfg: Optional[Any] = None):
        self.cfg = cfg or type("C", (), {})()
        # defaults
        self.n_bins = getattr(self.cfg, "n_bins", 20)
        self.mmd_gamma = getattr(self.cfg, "mmd_gamma", None)
        self.random_state = getattr(self.cfg, "random_state", 42)

        self.rf = RandomForestClassifier(
            n_estimators=getattr(self.cfg, "rf_n_estimators", 120),
            max_depth=getattr(self.cfg, "rf_max_depth", 8),
            random_state=self.random_state,
            n_jobs=getattr(self.cfg, "rf_n_jobs", -1)
        )

        self.xgb = None
        if XGBClassifier is not None and getattr(self.cfg, "use_xgb", True):
            try:
                self.xgb = XGBClassifier(
                    n_estimators=getattr(self.cfg, "xgb_n_estimators", 150),
                    max_depth=getattr(self.cfg, "xgb_max_depth", 6),
                    learning_rate=getattr(self.cfg, "xgb_lr", 0.1),
                    use_label_encoder=False,
                    eval_metric="logloss",
                    random_state=self.random_state,
                )
            except Exception:
                self.xgb = None

        self.lgb = None
        if LGBMClassifier is not None and getattr(self.cfg, "use_lgb", True):
            try:
                self.lgb = LGBMClassifier(
                    n_estimators=getattr(self.cfg, "lgb_n_estimators", 150),
                    max_depth=getattr(self.cfg, "lgb_max_depth", 6),
                    random_state=self.random_state,
                )
            except Exception:
                self.lgb = None

    # -----------------
    # Distribution (KL)
    # -----------------
    def distribution_score(self, real_df: pd.DataFrame, fake_df: pd.DataFrame) -> float:
        """
        For each numeric column compute KL(real || fake) using shared bins,
        convert to similarity score via exp(-kl) and return mean similarity.
        """
        real_df, fake_df, cols = _safe_numeric_df(real_df, fake_df)
        if len(cols) == 0:
            return 0.0

        scores = []
        for c in cols:
            try:
                kl = _shared_hist_kl(real_df[c].values, fake_df[c].values, bins=self.n_bins)
                if np.isnan(kl):
                    continue
                scores.append(float(np.exp(-kl)))  # 1 means identical
            except Exception:
                continue
        return float(np.mean(scores)) if scores else 0.0

    # -----------------
    # MMD (RBF)
    # -----------------
    def mmd_score(self, real_df: pd.DataFrame, fake_df: pd.DataFrame) -> float:
        """
        Multi-dimensional MMD: compute MMD on the numeric matrix.
        Return sqrt(mmd2) to be comparable on scale.
        Lower -> better (0 = identical)
        """
        real_df, fake_df, cols = _safe_numeric_df(real_df, fake_df)
        if len(cols) == 0:
            return float("nan")

        X = real_df.values
        Y = fake_df.values
        mmd2 = _mmd_rbf(X, Y, gamma=self.mmd_gamma)
        return float(np.sqrt(mmd2))

    # -----------------
    # Wasserstein average per feature
    # -----------------
    def wasserstein_score(self, real_df: pd.DataFrame, fake_df: pd.DataFrame) -> float:
        """
        Compute mean 1D Wasserstein distance across numeric columns.
        Lower -> better.
        """
        real_df, fake_df, cols = _safe_numeric_df(real_df, fake_df)
        if len(cols) == 0:
            return float("nan")

        dists = []
        for c in cols:
            try:
                u = real_df[c].values
                v = fake_df[c].values
                d = _wasserstein_1d(u, v)
                dists.append(float(d))
            except Exception:
                continue
        return float(np.mean(dists)) if dists else float("nan")

    # -----------------
    # Frechet distance
    # -----------------
    def frechet_score(self, real_df: pd.DataFrame, fake_df: pd.DataFrame) -> float:
        """
        Frechet distance between multivariate Gaussians fitted on real and fake numeric matrices.
        Lower -> better.
        """
        real_df, fake_df, cols = _safe_numeric_df(real_df, fake_df)
        if len(cols) == 0:
            return float("nan")
        X = real_df.values
        Y = fake_df.values
        try:
            return float(_frechet_distance(X, Y))
        except Exception:
            return float("nan")

    # -----------------
    # Classifier-based AUC (RF / XGB / LGB)
    # -----------------
    def _classifier_auc(self, clf, X, y):
        """
        Fit classifier on training split and return AUC on test split.
        We fit scaler on training set only.
        Returns 1 - AUC so higher = more indistinguishable.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        if X.shape[0] < 4 or len(np.unique(y)) < 2:
            return float("nan")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y if len(np.unique(y)) > 1 else None
        )
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        try:
            clf.fit(X_train_s, y_train)
            proba = clf.predict_proba(X_test_s)[:, 1]
            auc = roc_auc_score(y_test, proba)
            # Return 1 - AUC so that higher means better (indistinguishable)
            return float(1.0 - auc)
        except Exception:
            return float("nan")

    def rf_score(self, real_df: pd.DataFrame, fake_df: pd.DataFrame) -> float:
        real_df, fake_df, cols = _safe_numeric_df(real_df, fake_df)
        if len(cols) == 0:
            return float("nan")

        # stack and label
        r = real_df.copy()
        f = fake_df.copy()
        r["__label__"] = 1
        f["__label__"] = 0
        df = pd.concat([r, f], axis=0).reset_index(drop=True)
        y = df["__label__"].values
        X = df[cols].values

        return self._classifier_auc(self.rf, X, y)

    def xgb_score(self, real_df: pd.DataFrame, fake_df: pd.DataFrame) -> float:
        if self.xgb is None:
            return float("nan")
        real_df, fake_df, cols = _safe_numeric_df(real_df, fake_df)
        if len(cols) == 0:
            return float("nan")
        r = real_df.copy()
        f = fake_df.copy()
        r["__label__"] = 1
        f["__label__"] = 0
        df = pd.concat([r, f], axis=0).reset_index(drop=True)
        y = df["__label__"].values
        X = df[cols].values
        return self._classifier_auc(self.xgb, X, y)

    def lgb_score(self, real_df: pd.DataFrame, fake_df: pd.DataFrame) -> float:
        if self.lgb is None:
            return float("nan")
        real_df, fake_df, cols = _safe_numeric_df(real_df, fake_df)
        if len(cols) == 0:
            return float("nan")
        r = real_df.copy()
        f = fake_df.copy()
        r["__label__"] = 1
        f["__label__"] = 0
        df = pd.concat([r, f], axis=0).reset_index(drop=True)
        y = df["__label__"].values
        X = df[cols].values
        return self._classifier_auc(self.lgb, X, y)

    def classifier_score(self, real_df: pd.DataFrame, fake_df: pd.DataFrame, model="rf") -> float:
        """
        Compatibility dispatcher used by overall_score.
        model: "rf", "xgb", or "lgb"
        Returns the same 1-AUC measure as rf_score/xgb_score/lgb_score above.
        """
        if model == "rf":
            return self.rf_score(real_df, fake_df)
        elif model == "xgb":
            return self.xgb_score(real_df, fake_df)
        elif model == "lgb":
            return self.lgb_score(real_df, fake_df)
        else:
            return float("nan")

    # -----------------
    # Overall score aggregation
    # -----------------
    def overall_score(self, real_df: pd.DataFrame, fake_df: pd.DataFrame) -> Dict[str, Any]:
        """
        计算所有指标 + 以 Markdown 表格格式输出 + 结构化返回
        """
        # 1. 各指标
        dist = self.distribution_score(real_df, fake_df)
        mmd = self.mmd_score(real_df, fake_df)
        wass = self.wasserstein_score(real_df, fake_df)
        fre = self.frechet_score(real_df, fake_df)

        auc_rf = self.classifier_score(real_df, fake_df, model="rf")
        auc_xgb = self.classifier_score(real_df, fake_df, model="xgb")
        auc_lgb = self.classifier_score(real_df, fake_df, model="lgb")

        # 2. 整合
        data = {
            "Metric": [
                "Distribution Similarity (exp(-KL))",
                "MMD (RBF, lower=better)",
                "Wasserstein (1D avg, lower=better)",
                "Frechet Distance (FID-like, lower=better)",
                "1-AUC RandomForest (higher=better)",
                "1-AUC XGBoost (higher=better)",
                "1-AUC LightGBM (higher=better)",
            ],
            "Score": [
                dist,
                mmd,
                wass,
                fre,
                auc_rf,
                auc_xgb,
                auc_lgb,
            ]
        }

        df = pd.DataFrame(data)

        # 3. 生成 markdown 输出
        try:
            markdown_table = df.to_markdown(index=False)
        except Exception:
            # pandas older versions might not have to_markdown
            markdown_table = df.to_string(index=False)

        logger.info("\n===== Job Quality Evaluation Result =====\n" + markdown_table)

        # 4. 返回结构化数据 + markdown
        return {
            "distribution": dist,
            "mmd": mmd,
            "wasserstein": wass,
            "frechet": fre,
            "auc_rf": auc_rf,
            "auc_xgb": auc_xgb,
            "auc_lgb": auc_lgb,
            "markdown": markdown_table
        }

    # per-feature KL Bar Chart（柱状图）
    def plot_kl_bars(self, real_df, fake_df, save_path=None):
        r, f, cols = _safe_numeric_df(real_df, fake_df)
        if len(cols) == 0:
            logger.warning("No numeric columns to plot KL bars.")
            return

        kl_values = []
        for c in cols:
            try:
                kl = _shared_hist_kl(r[c].values, f[c].values)
                kl_values.append(kl)
            except Exception:
                kl_values.append(np.nan)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(max(8, len(cols) * 0.5), 5))
        plt.bar(cols, [0 if np.isnan(x) else x for x in kl_values])
        plt.xticks(rotation=45, ha='right')
        plt.title("Per-Feature KL Divergence")
        plt.ylabel("KL(real || fake)")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"Saved KL bar chart to {save_path}")
        plt.close()

    # MMD Kernel Heatmap（核矩阵热力图）
    def plot_mmd_kernel(self, real_df, fake_df, save_path=None):
        r, f, cols = _safe_numeric_df(real_df, fake_df)
        if len(cols) == 0:
            logger.warning("No numeric columns to plot MMD kernel.")
            return

        X = r.values
        Y = f.values

        K = _rbf_kernel(X, Y, gamma=self.mmd_gamma)

        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(6, 5))
        sns.heatmap(K, cmap="viridis")
        plt.title("RBF Kernel Matrix between Real & Fake")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"Saved MMD kernel heatmap to {save_path}")
        plt.close()

    # Classifier ROC Curves（分类器 ROC 曲线）
    def plot_roc(self, real_df, fake_df, model="rf", save_path=None):
        r, f, cols = _safe_numeric_df(real_df, fake_df)
        if len(cols) == 0:
            logger.warning("No numeric columns to plot ROC.")
            return

        X_df = pd.concat([r, f], axis=0).reset_index(drop=True)
        y = np.concatenate([np.ones(len(r)), np.zeros(len(f))])

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_df, y, test_size=0.25, random_state=self.random_state, stratify=y if len(np.unique(y)) > 1 else None
            )
        except Exception:
            # fallback without stratify
            X_train, X_test, y_train, y_test = train_test_split(
                X_df, y, test_size=0.25, random_state=self.random_state
            )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if model == "rf":
            clf = self.rf
        elif model == "xgb":
            clf = self.xgb
        elif model == "lgb":
            clf = self.lgb
        else:
            logger.error(f"Unknown model {model} for ROC plot.")
            return

        if clf is None:
            logger.warning(f"Model {model} not available for ROC plotting.")
            return

        try:
            clf.fit(X_train, y_train)
            prob = clf.predict_proba(X_test)[:, 1]
        except Exception as e:
            logger.error(f"Classifier training failed for ROC: {e}")
            return

        from sklearn.metrics import roc_curve, auc

        fpr, tpr, _ = roc_curve(y_test, prob)
        roc_auc = auc(fpr, tpr)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"ROC {model.upper()} (AUC={roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve ({model.upper()})")
        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"Saved ROC curve to {save_path}")
        plt.close()
