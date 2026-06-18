import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_selection import (
    f_regression,
    mutual_info_regression,
    RFE,
    SequentialFeatureSelector,
)
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor


def _feature_names_to_indices(names, all_names):
    return [all_names.index(n) for n in names]


def rank_by_pearson(X_train, y_train):
    y_train_arr = y_train.to_numpy() if hasattr(y_train, "to_numpy") else np.array(y_train)
    corr_values = []
    for col in X_train.columns:
        corr = np.corrcoef(X_train[col], y_train_arr)[0, 1]
        corr_values.append(abs(corr))
    order = np.argsort(corr_values)[::-1]
    return [X_train.columns[i] for i in order]


def rank_by_spearman(X_train, y_train):
    y_train_arr = y_train.to_numpy() if hasattr(y_train, "to_numpy") else np.array(y_train)
    corr_values = []
    for col in X_train.columns:
        rho, _ = spearmanr(X_train[col], y_train_arr)
        corr_values.append(abs(rho))
    order = np.argsort(corr_values)[::-1]
    return [X_train.columns[i] for i in order]


def rank_by_f_test(X_train, y_train):
    f_scores, _ = f_regression(X_train, y_train)
    order = np.argsort(f_scores)[::-1]
    return [X_train.columns[i] for i in order]


def rank_by_mutual_info(X_train, y_train, random_state=42):
    mi_scores = mutual_info_regression(X_train, y_train, random_state=random_state)
    order = np.argsort(mi_scores)[::-1]
    return [X_train.columns[i] for i in order]


def rank_by_rfe(X_train, y_train):
    estimator = LinearRegression()
    selector = RFE(estimator, n_features_to_select=1, step=1)
    selector.fit(X_train, y_train)
    rankings = selector.ranking_
    order = np.argsort(rankings)
    return [X_train.columns[i] for i in order]


def rank_by_sfs_forward(X_train, y_train):
    features = list(X_train.columns)
    remaining = set(features)
    selected = []
    y_train_arr = y_train.to_numpy() if hasattr(y_train, "to_numpy") else y_train.values.ravel()

    while remaining:
        best_score = -np.inf
        best_feat = None
        for feat in sorted(remaining):
            current_feats = selected + [feat]
            model = LinearRegression()
            model.fit(X_train[current_feats], y_train_arr)
            score = model.score(X_train[current_feats], y_train_arr)
            if score > best_score:
                best_score = score
                best_feat = feat
        selected.append(best_feat)
        remaining.remove(best_feat)

    return selected


def rank_by_sbs_backward(X_train, y_train):
    features = list(X_train.columns)
    y_train_arr = y_train.to_numpy() if hasattr(y_train, "to_numpy") else y_train.values.ravel()

    current = set(features)
    removed_order = []

    while len(current) > 1:
        best_score = -np.inf
        worst_feat = None
        current_list = sorted(current)
        for feat in current_list:
            subset = [f for f in current_list if f != feat]
            model = LinearRegression()
            model.fit(X_train[subset], y_train_arr)
            score = model.score(X_train[subset], y_train_arr)
            if score > best_score:
                best_score = score
                worst_feat = feat
        removed_order.append(worst_feat)
        current.remove(worst_feat)

    last_feature = list(current)[0]
    removed_order.append(last_feature)

    return list(reversed(removed_order))


def rank_by_lasso(X_train, y_train, random_state=42):
    model = LassoCV(cv=5, random_state=random_state, max_iter=10000)
    model.fit(X_train, y_train)
    coef_abs = np.abs(model.coef_)
    order = np.argsort(coef_abs)[::-1]
    return [X_train.columns[i] for i in order]


def rank_by_random_forest(X_train, y_train, random_state=42):
    model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    return [X_train.columns[i] for i in order]


def get_all_feature_rankings(X_train, y_train) -> "dict[str, list[str]]":
    rankings = {
        "Pearson": rank_by_pearson(X_train, y_train),
        "Spearman": rank_by_spearman(X_train, y_train),
        "F-test": rank_by_f_test(X_train, y_train),
        "Mutual Info": rank_by_mutual_info(X_train, y_train),
        "RFE": rank_by_rfe(X_train, y_train),
        "SFS (Fwd)": rank_by_sfs_forward(X_train, y_train),
        "SBS (Bwd)": rank_by_sbs_backward(X_train, y_train),
        "Lasso (L1)": rank_by_lasso(X_train, y_train),
        "Random Forest": rank_by_random_forest(X_train, y_train),
    }
    return rankings
