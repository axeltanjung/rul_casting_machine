import argparse
import pathlib

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
import joblib


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ccm_rul_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "rul_gbr_pipeline.joblib"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate the key feature engineering from the modeling baseline notebook."""
    df = df.copy()

    # Keep only rows with RUL
    df = df.dropna(subset=["RUL"])

    group_cols = ["num_crystallizer", "num_stream"]

    # Physical deltas
    df["steel_weight_error"] = (
        df["steel_weight, tonn"] - df["steel_weight_theoretical, tonn"]
    )

    df["total_residuals"] = (
        df["slag_weight_close_grab1, tonn"].fillna(0)
        + df["metal_residue_grab1, tonn"].fillna(0)
        + df["residuals_grab2, tonn"].fillna(0)
    )

    df["temp_delta_measurement"] = (
        df["temperature_measurement2, Celsius deg."]
        - df["temperature_measurement1, Celsius deg."]
    )

    df["cooling_efficiency"] = (
        df["water_temperature_delta, Celsius deg."]
        / df["water_consumption, liter/minute"].replace(0, np.nan)
    )

    # Rolling statistics
    roll_cols = [
        "steel_temperature_grab1, Celsius deg.",
        "water_temperature_delta, Celsius deg.",
        "alloy_speed, meter/minute",
        "swing_frequency, amount/minute",
        "resistance, tonn",
    ]
    roll_windows = [3, 5]

    for col in roll_cols:
        for w in roll_windows:
            df[f"{col}_roll_mean_{w}"] = (
                df.groupby(group_cols)[col]
                .transform(lambda x: x.rolling(w, min_periods=1).mean())
            )
            df[f"{col}_roll_std_{w}"] = (
                df.groupby(group_cols)[col]
                .transform(lambda x: x.rolling(w, min_periods=1).std())
            )

    # Finite difference (trend)
    for col in [
        "steel_temperature_grab1, Celsius deg.",
        "water_temperature_delta, Celsius deg.",
        "alloy_speed, meter/minute",
    ]:
        df[f"{col}_diff"] = df.groupby(group_cols)[col].diff()

    # Chemistry aggregation
    chem_cols = [c for c in df.columns if c.endswith(", %")]
    if chem_cols:
        df["chem_sum"] = df[chem_cols].sum(axis=1)
        df["chem_std"] = df[chem_cols].std(axis=1)

    return df


def asymmetric_rul_error(y_true: np.ndarray, y_pred: np.ndarray, over_penalty: float = 2.0) -> float:
    """Asymmetric error: penalize overestimation of RUL more than underestimation."""
    err = y_pred - y_true
    return float(np.mean(np.where(err > 0, over_penalty * err, np.abs(err))))


def train_and_evaluate(df: pd.DataFrame) -> Pipeline:
    target = "RUL"
    drop_cols = ["date"]

    X = df.drop(columns=[target] + [c for c in drop_cols if c in df.columns])
    y = df[target]

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("var", VarianceThreshold(threshold=1e-4)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ],
        remainder="drop",
    )

    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    # Time-series aware split for validation
    tscv = TimeSeriesSplit(n_splits=5)
    train_idx, val_idx = list(tscv.split(X))[-1]

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)

    mae = mean_absolute_error(y_val, y_pred)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    asym = asymmetric_rul_error(y_val.values, y_pred)

    print("Validation metrics")
    print(f"  MAE : {mae:,.3f}")
    print(f"  RMSE: {rmse:,.3f}")
    print(f"  Asym: {asym:,.3f}")

    # Retrain on full data before saving
    pipeline.fit(X, y)
    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RUL prediction model.")
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(DATA_PATH),
        help="Path to ccm_rul_dataset.csv",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(MODEL_PATH),
        help="Where to save the trained model pipeline.",
    )
    args = parser.parse_args()

    data_path = pathlib.Path(args.data_path)
    output_path = pathlib.Path(args.output_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns.")

    df_feat = engineer_features(df)
    print(f"After feature engineering: {len(df_feat):,} rows, {len(df_feat.columns)} columns.")

    pipeline = train_and_evaluate(df_feat)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    print(f"Saved trained RUL model pipeline to: {output_path}")


if __name__ == "__main__":
    main()


