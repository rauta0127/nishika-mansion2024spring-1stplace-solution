import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import pickle
import numpy as np
import warnings

warnings.simplefilter("ignore")

INPUT_DIR = "input"
OUTPUT_DIR = "output/train"
PREPROCESS_OUTPUT_DIR = "output/preprocess"
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


def train(df):
    feats = [c for c in df.columns if not c in IGNORE_FEATS]
    print(f"The Number of Features = {len(feats)}")
    train_df = df[df["test"] == 0].reset_index(drop=True)
    train_df[f"preds_{TARGET_COL}"] = 0

    feature_importance_df_list = []

    for seed in SEEDS:

        train_df["fold"] = 0

        col = "prefecture"
        kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for i, (train_idx, valid_idx) in enumerate(kf.split(train_df, train_df[col])):
            train_df.loc[valid_idx, "fold"] = i

        train_df[f"preds_{TARGET_COL}_{seed}"] = 0

        for i in range(N_SPLITS):
            valid_idx = train_df[train_df["fold"] == i].index
            train_idx = train_df[train_df["fold"] != i].index
            valid_fold_df = train_df.loc[valid_idx, :].reset_index(drop=True)
            train_fold_df = train_df.loc[train_idx, :].reset_index(drop=True)

            train_x, train_y = train_fold_df[feats], train_fold_df[TARGET_COL]
            valid_x, valid_y = valid_fold_df[feats], valid_fold_df[TARGET_COL]

            train_params = {
                "boosting_type": "gbdt",
                "objective": "huber",
                "alpha": 0.097,
                "metric": "mae",
                "verbose": -1,
                "random_sate": seed,
                "lambda_l1": 0.3,
                "lambda_l2": 0.01,
            }

            lgb_train = lgb.Dataset(train_x, train_y)
            lgb_valid = lgb.Dataset(valid_x, valid_y)

            print(f"training start...")
            model = lgb.train(
                train_params,
                lgb_train,
                valid_sets=lgb_valid,
                num_boost_round=50000,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=300, verbose=True),
                    lgb.log_evaluation(1000),
                ],
            )
            print(f"training end.")
            model_path = f"{OUTPUT_DIR}/model_seed{seed}_{i}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            valid_preds = model.predict(valid_x)

            y_true = valid_y.values
            y_pred = valid_preds
            valid_score = mean_absolute_error(y_true, y_pred)
            print(f"valid_score = {valid_score}")

            train_df.loc[valid_idx, f"preds_{TARGET_COL}_{seed}"] += valid_preds

            feature_importance_df = sorted(
                zip(
                    model.feature_name(),
                    model.feature_importance(importance_type="gain"),
                ),
                key=lambda x: x[1],
                reverse=True,
            )
            feature_importance_df = pd.DataFrame(feature_importance_df).rename(
                columns={0: "feature", 1: "importance_gain"}
            )
            feature_importance_df = feature_importance_df.sort_values(
                "importance_gain", ascending=False
            ).reset_index(drop=True)
            feature_importance_df["importance_normalized"] = (
                feature_importance_df["importance_gain"]
                / feature_importance_df["importance_gain"].sum()
            )
            feature_importance_df["fold"] = i
            feature_importance_df_list.append(feature_importance_df)

        train_df[f"preds_{TARGET_COL}"] += train_df[
            f"preds_{TARGET_COL}_{seed}"
        ].values / len(SEEDS)

    validation_df = train_df[
        ["id", "land_area", "transaction_price_total_log", f"preds_{TARGET_COL}"]
    ]
    feature_importance_df = pd.concat(feature_importance_df_list).reset_index(drop=True)
    feature_importance_df = (
        feature_importance_df.groupby("feature")["importance_normalized"]
        .mean()
        .reset_index()
    )
    feature_importance_df = feature_importance_df.sort_values(
        "importance_normalized", ascending=False
    ).reset_index(drop=True)
    feature_importance_df.to_csv(f"{OUTPUT_DIR}/feature_importance_df.csv", index=False)
    return validation_df


def evaluate(preds_df):
    eval_df = preds_df.copy()
    eval_df["preds_transaction_price_total"] = (
        np.expm1(eval_df[f"preds_{TARGET_COL}"]) * eval_df["land_area"]
    )
    eval_df["preds_transaction_price_total_log"] = np.log10(
        eval_df["preds_transaction_price_total"]
    )
    y_true = eval_df["transaction_price_total_log"].values
    y_pred = eval_df["preds_transaction_price_total_log"].values
    validation_score = mean_absolute_error(y_true, y_pred)
    print(f"validation_score = {validation_score}")
    return validation_score


if __name__ == "__main__":
    df = pd.read_csv(f"{PREPROCESS_OUTPUT_DIR}/df.csv")
    validation_df = train(df)
    validation_df.to_csv(f"{OUTPUT_DIR}/validation_df.csv", index=False)
    validation_score = evaluate(validation_df)
