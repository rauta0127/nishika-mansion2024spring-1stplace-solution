import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import pickle
import numpy as np
import warnings
from datetime import datetime

warnings.simplefilter("ignore")

INPUT_DIR = "input"
OUTPUT_DIR = "output/predict"
PREPROCESS_OUTPUT_DIR = "output/preprocess"
TRAIN_OUTPUT_DIR = "output/train"
TARGET_COL = "transaction_price_per_area_log"
IGNORE_FEATS = [
    TARGET_COL,
    "transaction_price_total_log",
    "id",
    "test",
    "address",
    "station",
    "transaction_price_total",
    "transaction_price_per_area",
]
N_SPLITS = 5
SEEDS = [2024, 2025, 2026]


def predict(df):
    feats = [c for c in df.columns if not c in IGNORE_FEATS]
    test_df = df[df["test"] == 1].reset_index(drop=True)
    test_df[f"preds_{TARGET_COL}"] = 0

    for seed in SEEDS:

        test_df[f"preds_{TARGET_COL}_{seed}"] = 0

        for i in range(N_SPLITS):
            test_x = test_df[feats]
            model_path = f"{TRAIN_OUTPUT_DIR}/model_seed{seed}_{i}.pkl"
            with open(model_path, "rb") as f:
                model = pickle.load(f)

            test_preds = model.predict(test_x)
            test_df.loc[:, f"preds_{TARGET_COL}_{seed}"] += test_preds / N_SPLITS

        test_df[f"preds_{TARGET_COL}"] += test_df[
            f"preds_{TARGET_COL}_{seed}"
        ].values / len(SEEDS)

    preds_df = test_df[
        ["id", "land_area", "transaction_price_total_log", f"preds_{TARGET_COL}"]
    ]
    preds_df["preds_transaction_price_total"] = (
        np.expm1(preds_df[f"preds_{TARGET_COL}"]) * preds_df["land_area"]
    )
    preds_df["preds_transaction_price_total_log"] = np.log10(
        preds_df["preds_transaction_price_total"]
    )
    return preds_df


def create_submission(preds_df):
    subm_df = preds_df[["id", "preds_transaction_price_total_log"]].copy()
    subm_df = subm_df.rename(
        columns={
            "id": "ID",
            "preds_transaction_price_total_log": "取引価格（総額）_log",
        }
    )
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    subm_df.to_csv(f"{OUTPUT_DIR}/submission_{now}.csv", index=False)
    return subm_df


if __name__ == "__main__":
    df = pd.read_csv(f"{PREPROCESS_OUTPUT_DIR}/df.csv")
    preds_df = predict(df)
    preds_df.to_csv(f"{OUTPUT_DIR}/preds_df.csv", index=False)
    subm_df = create_submission(preds_df)
