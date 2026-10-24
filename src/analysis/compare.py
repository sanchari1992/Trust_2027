"""Compares two sets of questions (e.g. real queries vs. a generated benchmark):
  1. per-feature summary (mean, and whether the difference is statistically
     significant, via the Mann-Whitney U test, with a bootstrap confidence
     interval on the difference of means);
  2. a classifier two-sample test: can a simple model tell the two sets apart
     from these features alone? Its held-out AUC is a single, graph-independent
     number for "how different do these question sets look" (0.5 = indistinguishable,
     1.0 = perfectly separable). This is the check discussed for avoiding circularity
     when the SAME features later inform how questions get mutated.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from src.analysis.features import ALL_NUMERIC_FEATURES, REALISM_FEATURES


def bootstrap_mean_diff(a, b, n_samples, rng):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    diffs = [
        rng.choice(a, len(a), replace=True).mean() - rng.choice(b, len(b), replace=True).mean()
        for _ in range(n_samples)
    ]
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def feature_table(df_a, df_b, name_a, name_b, bootstrap_samples, seed):
    """One row per feature: means in each set, the difference, its 95% CI, and a
    p-value from the Mann-Whitney U test (does not assume a normal distribution)."""
    rng = np.random.default_rng(seed)
    rows = []
    for feat in ALL_NUMERIC_FEATURES:
        a, b = df_a[feat].astype(float), df_b[feat].astype(float)
        a_clean, b_clean = a.dropna(), b.dropna()
        if a_clean.nunique() <= 1 and b_clean.nunique() <= 1 and a_clean.mean() == b_clean.mean():
            p = float("nan")  # identical constant in both sets; the test is undefined
        else:
            p = stats.mannwhitneyu(a_clean, b_clean, alternative="two-sided").pvalue
        lo, hi = bootstrap_mean_diff(a, b, bootstrap_samples, rng)
        rows.append({
            "feature": feat, f"mean_{name_a}": a.mean(), f"mean_{name_b}": b.mean(),
            "diff": a.mean() - b.mean(), "diff_ci_low": lo, "diff_ci_high": hi, "p_value": p,
        })
    table = pd.DataFrame(rows).sort_values("p_value")
    # Benjamini-Hochberg correction: with many features tested at once, some will
    # look "significant" by chance; q_value adjusts for that.
    table["q_value"] = _benjamini_hochberg(table["p_value"].values)
    return table.reset_index(drop=True)


def _benjamini_hochberg(pvals):
    """Adjusts p-values for testing many features at once. NaN p-values (a feature
    that is constant in both sets, so the test is undefined) are left as NaN and
    excluded from the correction, instead of turning every other q-value into NaN."""
    pvals = np.asarray(pvals, dtype=float)
    out = np.full(pvals.shape, np.nan)
    mask = ~np.isnan(pvals)
    p = pvals[mask]
    n = len(p)
    if n == 0:
        return out
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(n)
    adjusted[order] = np.clip(ranked, 0, 1)
    out[mask] = adjusted
    return out


def classifier_test(df_a, df_b, test_size, cv_folds, seed, features=None):
    """Can a simple classifier tell set A from set B? Default features: every feature except pure length (REALISM_FEATURES)."""
    features = list(features or REALISM_FEATURES)
    X = pd.concat([df_a[features], df_b[features]], ignore_index=True)
    X = X.fillna(X.median(numeric_only=True))
    y = np.array([1] * len(df_a) + [0] * len(df_b))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    cv_scores = []
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    for tr, va in skf.split(X_train_s, y_train):
        clf = LogisticRegression(max_iter=2000).fit(X_train_s[tr], y_train[tr])
        cv_scores.append(roc_auc_score(y_train[va], clf.predict_proba(X_train_s[va])[:, 1]))

    clf = LogisticRegression(max_iter=2000).fit(X_train_s, y_train)
    held_out_auc = roc_auc_score(y_test, clf.predict_proba(X_test_s)[:, 1])
    importance = dict(zip(features, clf.coef_[0]))
    top = sorted(importance.items(), key=lambda kv: -abs(kv[1]))[:8]
    return {
        "held_out_auc": float(held_out_auc),
        "cv_auc_mean": float(np.mean(cv_scores)),
        "cv_auc_std": float(np.std(cv_scores)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "top_features_by_weight": [{"feature": f, "weight": float(w)} for f, w in top],
    }
