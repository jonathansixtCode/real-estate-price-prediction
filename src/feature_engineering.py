"""
Module 4: Feature Engineering, Transformation & Selection
Ames Housing Dataset — Leak-Free Feature Pipeline

Provides modular functions for:
- Constructing domain-specific features without data leakage
- Applying log1p target/skew transforms, ordinal encoding, one-hot encoding, and scaling
- Reducing multicollinearity (correlation + VIF) and selecting top features
- Exporting processed CSV matrices directly to `data/processed`
"""

from typing import Dict, List, Tuple, Optional
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


ORDINAL_QUALITY_MAP: Dict[str, int] = {
    "None": 0,
    "NA": 0,
    "Po": 1,
    "Fa": 2,
    "TA": 3,
    "Gd": 4,
    "Ex": 5,
}

BASEMENT_EXPOSURE_MAP: Dict[str, int] = {
    "None": 0,
    "NA": 0,
    "No": 1,
    "Mn": 2,
    "Av": 3,
    "Gd": 4,
}

BASEMENT_FINISH_MAP: Dict[str, int] = {
    "None": 0,
    "NA": 0,
    "Unf": 1,
    "LwQ": 2,
    "Rec": 3,
    "BLQ": 4,
    "ALQ": 5,
    "GLQ": 6,
}

GARAGE_FINISH_MAP: Dict[str, int] = {
    "None": 0,
    "NA": 0,
    "Unf": 1,
    "RFn": 2,
    "Fin": 3,
}


def construct_derived_features(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    target_col: str = "SalePrice",
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Create domain-engineered features without data leakage.
    Neighborhood price-per-sq-meter is computed strictly from `train_df`.
    """
    train_out = train_df.copy()
    test_out = test_df.copy() if test_df is not None else None

    if target_col in train_out.columns and "Neighborhood" in train_out.columns:
        gr_liv = train_out["GrLivArea"].replace(0, np.nan) if "GrLivArea" in train_out.columns else None
        if gr_liv is not None:
            sqm_area = gr_liv * 0.092903
            price_per_sqm = train_out[target_col] / sqm_area
            nbhd_medians = price_per_sqm.groupby(train_out["Neighborhood"]).median()
            global_median = float(price_per_sqm.median())
        else:
            nbhd_medians = pd.Series(dtype=float)
            global_median = 1296.0
    else:
        nbhd_medians = pd.Series(dtype=float)
        global_median = 1296.0

    def _apply_domain_features(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        bsmt_sf = out["TotalBsmtSF"].fillna(0) if "TotalBsmtSF" in out.columns else 0
        gr_liv_sf = out["GrLivArea"].fillna(0) if "GrLivArea" in out.columns else 0
        out["TotalSF"] = gr_liv_sf + bsmt_sf
        out["GrLivArea_SqM"] = gr_liv_sf * 0.092903
        out["TotalSF_SqM"] = out["TotalSF"] * 0.092903

        if "Neighborhood" in out.columns and len(nbhd_medians) > 0:
            out["Neighborhood_Median_PricePerSqM"] = (
                out["Neighborhood"].map(nbhd_medians).fillna(global_median)
            )

        yr_sold = out["YrSold"] if "YrSold" in out.columns else 2010
        if "YearBuilt" in out.columns:
            out["PropertyAge"] = (yr_sold - out["YearBuilt"]).clip(lower=0)
            out["IsNewBuild"] = (out["PropertyAge"] <= 2).astype(int)
        if "YearRemodAdd" in out.columns:
            out["YearsSinceRemodel"] = (yr_sold - out["YearRemodAdd"]).clip(lower=0)
            out["IsRenovated"] = (
                (out["YearRemodAdd"] != out["YearBuilt"]).astype(int)
                if "YearBuilt" in out.columns
                else 0
            )
            if "YearBuilt" in out.columns:
                out["RemodelSpanYears"] = (out["YearRemodAdd"] - out["YearBuilt"]).clip(lower=0)
        if "GarageYrBlt" in out.columns:
            garage_yr = out["GarageYrBlt"].fillna(out["YearBuilt"] if "YearBuilt" in out.columns else yr_sold)
            out["GarageAge"] = (yr_sold - garage_yr).clip(lower=0)

        full_b = out["FullBath"].fillna(0) if "FullBath" in out.columns else 0
        half_b = out["HalfBath"].fillna(0) if "HalfBath" in out.columns else 0
        bsmt_full = out["BsmtFullBath"].fillna(0) if "BsmtFullBath" in out.columns else 0
        bsmt_half = out["BsmtHalfBath"].fillna(0) if "BsmtHalfBath" in out.columns else 0
        out["TotalBathrooms"] = full_b + 0.5 * half_b + bsmt_full + 0.5 * bsmt_half

        porch_cols = [
            c for c in ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "WoodDeckSF"]
            if c in out.columns
        ]
        out["TotalPorchSF"] = out[porch_cols].fillna(0).sum(axis=1) if porch_cols else 0.0

        if "LotArea" in out.columns:
            lot_safe = out["LotArea"].replace(0, np.nan)
            out["LivingToLotRatio"] = (gr_liv_sf / lot_safe).fillna(0.0)

        amenity_cols = [c for c in ["Fireplaces", "GarageCars", "TotalBathrooms"] if c in out.columns]
        out["AmenityProximityScore"] = out[amenity_cols].fillna(0).sum(axis=1) if amenity_cols else 0.0

        if "OverallQual" in out.columns:
            out["Qual_x_TotalSF"] = out["OverallQual"].fillna(0) * out["TotalSF"]
            out["Qual_x_GrLivArea"] = out["OverallQual"].fillna(0) * gr_liv_sf

        return out

    train_engineered = _apply_domain_features(train_out)
    test_engineered = _apply_domain_features(test_out) if test_out is not None else None
    return train_engineered, test_engineered


def encode_and_transform_features(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    target_col: str = "SalePrice",
    skew_threshold: float = 0.75,
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame]]:
    """
    Transform target with log1p, encode ordinal and nominal features without leakage,
    unskew skewed numeric predictors, and scale with RobustScaler fit on train only.
    """
    train_work = train_df.copy()
    test_work = test_df.copy() if test_df is not None else None

    if target_col not in train_work.columns:
        raise ValueError(f"Target column '{target_col}' not found in training data.")

    y_train_log = pd.Series(
        np.log1p(train_work[target_col].astype(float)),
        index=train_work.index,
        name=f"{target_col}_log",
    )

    drop_cols = [c for c in [target_col, "Id"] if c in train_work.columns]
    X_tr = train_work.drop(columns=drop_cols)
    X_te = test_work.drop(columns=[c for c in ["Id"] if c in test_work.columns]) if test_work is not None else None

    qual_like_cols = [
        c for c in X_tr.columns
        if any(token in c for token in ["Qual", "Cond", "Exter", "BsmtQual", "BsmtCond", "HeatingQC", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"])
        and X_tr[c].dtype == "object"
    ]
    for col in qual_like_cols:
        X_tr[col] = X_tr[col].fillna("None").map(ORDINAL_QUALITY_MAP).fillna(0)
        if X_te is not None and col in X_te.columns:
            X_te[col] = X_te[col].fillna("None").map(ORDINAL_QUALITY_MAP).fillna(0)

    if "BsmtExposure" in X_tr.columns and X_tr["BsmtExposure"].dtype == "object":
        X_tr["BsmtExposure"] = X_tr["BsmtExposure"].fillna("None").map(BASEMENT_EXPOSURE_MAP).fillna(0)
        if X_te is not None and "BsmtExposure" in X_te.columns:
            X_te["BsmtExposure"] = X_te["BsmtExposure"].fillna("None").map(BASEMENT_EXPOSURE_MAP).fillna(0)

    for bsmt_fin in ["BsmtFinType1", "BsmtFinType2"]:
        if bsmt_fin in X_tr.columns and X_tr[bsmt_fin].dtype == "object":
            X_tr[bsmt_fin] = X_tr[bsmt_fin].fillna("None").map(BASEMENT_FINISH_MAP).fillna(0)
            if X_te is not None and bsmt_fin in X_te.columns:
                X_te[bsmt_fin] = X_te[bsmt_fin].fillna("None").map(BASEMENT_FINISH_MAP).fillna(0)

    if "GarageFinish" in X_tr.columns and X_tr["GarageFinish"].dtype == "object":
        X_tr["GarageFinish"] = X_tr["GarageFinish"].fillna("None").map(GARAGE_FINISH_MAP).fillna(0)
        if X_te is not None and "GarageFinish" in X_te.columns:
            X_te["GarageFinish"] = X_te["GarageFinish"].fillna("None").map(GARAGE_FINISH_MAP).fillna(0)

    num_cols = X_tr.select_dtypes(include=[np.number]).columns.tolist()
    X_tr[num_cols] = X_tr[num_cols].fillna(X_tr[num_cols].median())
    if X_te is not None:
        te_num_cols = [c for c in num_cols if c in X_te.columns]
        X_te[te_num_cols] = X_te[te_num_cols].fillna(X_tr[te_num_cols].median())

    skewed_cols = [
        c for c in num_cols
        if X_tr[c].min() >= 0 and abs(float(X_tr[c].skew())) > skew_threshold
    ]
    for col in skewed_cols:
        X_tr[col] = np.log1p(X_tr[col])
        if X_te is not None and col in X_te.columns:
            X_te[col] = np.log1p(X_te[col].clip(lower=0))

    cat_cols = X_tr.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X_tr = pd.get_dummies(X_tr, columns=cat_cols, drop_first=True)
        if X_te is not None:
            te_cat_cols = [c for c in cat_cols if c in X_te.columns]
            X_te = pd.get_dummies(X_te, columns=te_cat_cols, drop_first=True)
            X_te = X_te.reindex(columns=X_tr.columns, fill_value=0)

    scaler = RobustScaler()
    scale_cols = X_tr.select_dtypes(include=[np.number]).columns.tolist()
    if scale_cols:
        X_tr[scale_cols] = scaler.fit_transform(X_tr[scale_cols])
        if X_te is not None:
            X_te[scale_cols] = scaler.transform(X_te[scale_cols])

    X_tr = X_tr.fillna(0.0)
    if X_te is not None:
        X_te = X_te.fillna(0.0)

    return X_tr, y_train_log, X_te


def select_features_pipeline(
    X_train: pd.DataFrame,
    y_train_log: pd.Series,
    X_test: Optional[pd.DataFrame] = None,
    corr_threshold: float = 0.90,
    vif_threshold: float = 10.0,
    max_vif_cols: int = 80,
    top_k: int = 60,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], List[str]]:
    """
    Prune extreme collinearity (|r| > corr_threshold and iterative VIF > vif_threshold)
    and select top_k features via target correlation rank.
    """
    X_tr = X_train.copy()
    X_te = X_test.copy() if X_test is not None else None

    if X_tr.shape[1] > 1:
        corr_matrix = X_tr.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_cols = [col for col in upper.columns if any(upper[col] > corr_threshold)]
        X_tr = X_tr.drop(columns=high_corr_cols, errors="ignore")

    if X_tr.shape[1] > 2:
        target_corr_all = X_tr.apply(lambda s: abs(s.corr(y_train_log))).fillna(0.0)
        vif_candidates = target_corr_all.sort_values(ascending=False).head(max_vif_cols).index.tolist()
        if len(vif_candidates) >= 2:
            sub_df = X_tr[vif_candidates].copy()
            while sub_df.shape[1] > 1:
                arr = sub_df.to_numpy(dtype=float)
                vifs = []
                for i in range(arr.shape[1]):
                    try:
                        v = variance_inflation_factor(arr, i)
                        vifs.append(np.inf if np.isnan(v) else v)
                    except Exception:
                        vifs.append(0.0)
                max_vif = max(vifs)
                if max_vif > vif_threshold:
                    drop_idx = int(np.argmax(vifs))
                    col_to_drop = sub_df.columns[drop_idx]
                    sub_df = sub_df.drop(columns=[col_to_drop])
                    X_tr = X_tr.drop(columns=[col_to_drop], errors="ignore")
                else:
                    break

    scores = X_tr.apply(lambda s: abs(s.corr(y_train_log))).fillna(0.0)
    selected_cols = scores.sort_values(ascending=False).head(min(top_k, X_tr.shape[1])).index.tolist()

    X_tr_selected = X_tr[selected_cols].copy()
    X_te_selected = None
    if X_te is not None:
        X_te_selected = X_te.reindex(columns=selected_cols, fill_value=0.0).copy()

    return X_tr_selected, X_te_selected, selected_cols


def save_processed_datasets(
    X_train_selected: pd.DataFrame,
    y_train_log: pd.Series,
    X_test_selected: Optional[pd.DataFrame] = None,
    output_dir: str = os.path.join("data", "processed"),
) -> Dict[str, str]:
    """
    Save engineered feature matrices and log-transformed target to CSV files
    directly inside the relative path `data/processed`.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths: Dict[str, str] = {}

    x_tr_path = os.path.join(output_dir, "X_train_selected.csv")
    y_tr_path = os.path.join(output_dir, "y_train_log.csv")
    X_train_selected.to_csv(x_tr_path, index=False)
    y_train_log.to_csv(y_tr_path, index=False)
    saved_paths["X_train_selected"] = x_tr_path
    saved_paths["y_train_log"] = y_tr_path

    if X_test_selected is not None:
        x_te_path = os.path.join(output_dir, "X_test_selected.csv")
        X_test_selected.to_csv(x_te_path, index=False)
        saved_paths["X_test_selected"] = x_te_path

    return saved_paths


def run_feature_engineering_pipeline(
    data_dir: str = os.path.join("data", "processed"),
    raw_dir: str = os.path.join("data", "raw"),
    target_col: str = "SalePrice",
    top_k: int = 60,
) -> Dict[str, str]:
    """
    End-to-end execution helper: loads cleaned data from `data/processed`
    (with fallback to `data/raw`), runs feature engineering & selection,
    and saves `X_train_selected.csv`, `y_train_log.csv`, and `X_test_selected.csv`
    directly into `data/processed`.
    """
    train_path = os.path.join(data_dir, "train_cleaned.csv")
    test_path = os.path.join(data_dir, "test_cleaned.csv")

    if not os.path.exists(train_path):
        train_path = os.path.join(raw_dir, "train.csv")
        test_path = os.path.join(raw_dir, "test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Neither {os.path.join(data_dir, 'train_cleaned.csv')} nor "
            f"{os.path.join(raw_dir, 'train.csv')} was found."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path) if os.path.exists(test_path) else None

    train_eng, test_eng = construct_derived_features(train_df, test_df, target_col=target_col)
    X_tr_enc, y_tr_log, X_te_enc = encode_and_transform_features(
        train_eng, test_eng, target_col=target_col
    )
    X_tr_sel, X_te_sel, _ = select_features_pipeline(
        X_tr_enc, y_tr_log, X_te_enc, top_k=top_k
    )

    return save_processed_datasets(
        X_train_selected=X_tr_sel,
        y_train_log=y_tr_log,
        X_test_selected=X_te_sel,
        output_dir=data_dir,
    )


if __name__ == "__main__":
    saved = run_feature_engineering_pipeline()
    for name, path in saved.items():
        print(f"Saved {name}: {path}")