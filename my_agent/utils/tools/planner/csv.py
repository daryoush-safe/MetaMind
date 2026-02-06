
import numpy as np
import pandas as pd
from typing import List, Optional, Any, Dict
from pydantic import Field, BaseModel
from langchain_core.tools import tool
from sklearn.model_selection import train_test_split


class DatasetSplitInput(BaseModel):
    file_path: str = Field(description="Path to the dataset file (CSV format)")
    target_column: str = Field(description="Name of the target column to be predicted")
    test_size: float = Field(default=0.2, ge=0.05, le=0.5, description="Proportion of the dataset reserved for testing")
    val_size: float = Field(default=0.1, ge=0.0, le=0.4, description="Proportion of the dataset reserved for validation")
    random_state: int = Field(default=42, ge=0, description="Random seed for reproducible train/validation/test splits")


def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        converted = [convert_numpy_types(item) for item in obj]
        return type(obj)(converted) if isinstance(obj, tuple) else converted
    else:
        return obj

@tool(args_schema=DatasetSplitInput)
def read_and_preprocess_csv(
    file_path: str,
    target_column: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Read a CSV file, preprocess it, and return train/val/test splits.

    Preprocessing steps performed:
    1. Handle missing values – numeric cols → median, categorical cols → mode.
       The 'Cabin' column (if present) is converted to a binary has_cabin flag.
    2. Encode categorical variables with one-hot / label encoding.
    3. Normalise numerical features to [0, 1] via min-max scaling.
    4. Split into train / validation / test sets.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.
    target_column : str
        Name of the target / label column.
    test_size : float
        Fraction held out for testing (default 0.2).
    val_size : float
        Fraction of the *remaining* data for validation (default 0.1).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        feature_names, n_samples, n_features, n_classes,
        class_distribution, X_train, y_train, X_val, y_val, X_test, y_test,
        preprocessing_log (list of strings describing what was done).
    """
    df = pd.read_csv(file_path)
    log: List[str] = []
    log.append(f"Loaded {len(df)} rows × {len(df.columns)} columns.")

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found. Available: {list(df.columns)}"}

    # ---- 1. Handle missing values ----
    # Special treatment for "Cabin" (common in Titanic-like datasets)
    if "Cabin" in df.columns:
        df["HasCabin"] = df["Cabin"].notna().astype(int)
        df.drop(columns=["Cabin"], inplace=True)
        log.append("Converted 'Cabin' → binary 'HasCabin'.")

    # Drop columns that are pure identifiers (unique per row)
    id_like = [c for c in df.columns if c != target_column and df[c].nunique() == len(df)]
    if id_like:
        df.drop(columns=id_like, inplace=True)
        log.append(f"Dropped identifier columns: {id_like}")

    # Drop columns with too many missing values (>50 %)
    high_null = [c for c in df.columns if df[c].isnull().mean() > 0.5 and c != target_column]
    if high_null:
        df.drop(columns=high_null, inplace=True)
        log.append(f"Dropped high-null columns (>50% missing): {high_null}")

    # Fill remaining missing values
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in ("float64", "float32", "int64", "int32"):
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                log.append(f"Filled '{col}' nulls with median={median_val:.4f}")
            else:
                mode_val = df[col].mode()[0] if not df[col].mode().empty else "UNKNOWN"
                df[col].fillna(mode_val, inplace=True)
                log.append(f"Filled '{col}' nulls with mode='{mode_val}'")

    # ---- 2. Encode categoricals ----
    cat_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns if c != target_column]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
        log.append(f"One-hot encoded: {cat_cols}")

    # Encode target if categorical
    target_is_categorical = df[target_column].dtype == "object" or df[target_column].dtype.name == "category"
    label_map = {}
    if target_is_categorical:
        labels = sorted(df[target_column].unique())
        label_map = {lbl: idx for idx, lbl in enumerate(labels)}
        df[target_column] = df[target_column].map(label_map)
        log.append(f"Label-encoded target: {label_map}")

    # ---- 3. Normalise numericals ----
    feature_cols = [c for c in df.columns if c != target_column]
    for col in feature_cols:
        cmin, cmax = df[col].min(), df[col].max()
        if cmax - cmin > 0:
            df[col] = (df[col] - cmin) / (cmax - cmin)
    log.append("Min-max normalised all feature columns to [0, 1].")

    # ---- 4. Train / val / test split ----

    X = df[feature_cols].values
    y = df[target_column].values

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(set(y)) > 1 else None,
    )
    actual_val = val_size / (1 - test_size)  # fraction of remaining
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=actual_val, random_state=random_state,
        stratify=y_temp if len(set(y_temp)) > 1 else None,
    )

    n_classes = len(set(y.tolist()))
    class_dist = {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}

    log.append(f"Split → train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    return convert_numpy_types({
        "feature_names": feature_cols,
        "n_samples": len(df),
        "n_features": len(feature_cols),
        "n_classes": n_classes,
        "class_distribution": class_dist,
        "label_map": label_map if label_map else None,
        "X_train": X_train.tolist(),
        "y_train": y_train.tolist(),
        "X_val": X_val.tolist(),
        "y_val": y_val.tolist(),
        "X_test": X_test.tolist(),
        "y_test": y_test.tolist(),
        "preprocessing_log": log,
    })
