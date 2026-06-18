import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler


def evaluate_feature_subset(X_train, X_test, y_train, y_test, selected_features, model=None):
    if model is None:
        model = LinearRegression()

    X_tr = X_train[selected_features]
    X_te = X_test[selected_features]

    y_tr = y_train.to_numpy().ravel() if hasattr(y_train, "to_numpy") else np.array(y_train).ravel()
    y_te = y_test.to_numpy().ravel() if hasattr(y_test, "to_numpy") else np.array(y_test).ravel()

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)

    model.fit(X_tr_scaled, y_tr)
    y_pred = model.predict(X_te_scaled)

    r2 = r2_score(y_te, y_pred)
    mse = mean_squared_error(y_te, y_pred)

    return {"r2": r2, "mse": mse, "features": selected_features}


def evaluate_all_methods_stepwise(X_train, X_test, y_train, y_test, rankings, max_k=13):
    results = {}

    for method_name, ranking in rankings.items():
        method_results = {"k": [], "r2": [], "mse": [], "features": []}
        for k in range(1, max_k + 1):
            top_features = ranking[:k]
            eval_result = evaluate_feature_subset(
                X_train, X_test, y_train, y_test, top_features
            )
            method_results["k"].append(k)
            method_results["r2"].append(eval_result["r2"])
            method_results["mse"].append(eval_result["mse"])
            method_results["features"].append(top_features)
        results[method_name] = method_results

    return results


def compute_best_frontier(results):
    methods = list(results.keys())
    max_k = len(results[methods[0]]["k"])
    frontier = {"k": list(range(1, max_k + 1)), "r2": [], "mse": []}

    for k_idx in range(max_k):
        r2_values = [results[m]["r2"][k_idx] for m in methods]
        mse_values = [results[m]["mse"][k_idx] for m in methods]
        frontier["r2"].append(max(r2_values))
        frontier["mse"].append(min(mse_values))

    return frontier
