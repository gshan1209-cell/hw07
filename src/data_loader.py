import pandas as pd


def load_boston_csv(path: str) -> "tuple[pd.DataFrame, pd.Series, list[str]]":
    df = pd.read_csv(path)
    df.columns = [col.lower() for col in df.columns]

    if "medv" not in df.columns:
        raise KeyError(
            f"Column 'medv' not found in CSV. Available columns: {df.columns.tolist()}"
        )

    target = df["medv"]
    features = df.drop(columns=["medv"])
    feature_names = features.columns.tolist()

    return features, target, feature_names
