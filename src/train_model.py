
import numpy as np
import pandas as pd
import joblib

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GroupShuffleSplit

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score
)


DATA_DIR = Path("data/ml")

MODEL_DIR = Path("models")

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


FEATURES = [

    "n_epochs",

    "stationary_rms",

    "linear_rms",

    "parallax_rms",

    "distance_au",

    "raw_parallax",

    "parallax_arcsec",

    "speed_arcsec_year",

    "circular_speed_ratio",

    "parallax_at_boundary",

    "mean_peak_snr",

    "min_peak_snr",

    "parallax_leverage",

    "improvement_over_stationary",

    "improvement_over_linear"

]


def evaluate_ranking(
    test,
    ranking_type
):

    results = []

    for simulation_id, group in test.groupby(
        "simulation_id"
    ):

        if ranking_type == "physics":

            ranked = group.sort_values(

                [
                    "passes_screen",
                    "n_epochs",
                    "physics_score"
                ],

                ascending=[
                    False,
                    False,
                    True
                ]
            )

        else:

            ranked = group.sort_values(

                "ml_score",

                ascending=False
            )

        labels = ranked[
            "label"
        ].to_numpy()

        correct_positions = (
            np.flatnonzero(labels == 1)
        )

        if len(correct_positions) > 0:

            best_rank = int(
                correct_positions[0] + 1
            )

        else:

            best_rank = np.inf

        results.append({

            "simulation_id":
                simulation_id,

            "method":
                ranking_type,

            "best_rank":
                best_rank,

            "top1":
                best_rank <= 1,

            "top5":
                best_rank <= 5,

            "top10":
                best_rank <= 10,

            "candidate_available":
                len(correct_positions) > 0

        })

    return pd.DataFrame(results)


def main():

    data = pd.read_csv(

        DATA_DIR / "training_data.csv"

    )

    data = data.replace(

        [np.inf, -np.inf],

        np.nan
    )

    data = data.dropna(

        subset=FEATURES + ["label"]

    ).reset_index(drop=True)

    X = data[FEATURES].astype(float)

    y = data["label"].astype(int)

    groups = data["simulation_id"]

    # Split by entire simulations,
    # not by individual candidate rows.

    splitter = GroupShuffleSplit(

        n_splits=1,

        test_size=0.25,

        random_state=42
    )

    train_idx, test_idx = next(

        splitter.split(
            X, y, groups
        )
    )

    X_train = X.iloc[train_idx]

    y_train = y.iloc[train_idx]

    X_test = X.iloc[test_idx]

    y_test = y.iloc[test_idx]

    if y_train.nunique() < 2:

        raise RuntimeError(

            "Training set requires both "
            "positive and negative examples. "
            "Generate more simulations."
        )

    if y_test.nunique() < 2:

        raise RuntimeError(

            "Test set requires both "
            "positive and negative examples. "
            "Generate more simulations."
        )

    print("TRAINING DATA")

    print(
        "Training simulations:",
        groups.iloc[train_idx].nunique()
    )

    print(
        "Testing simulations:",
        groups.iloc[test_idx].nunique()
    )

    print(
        "Training positives:",
        int(y_train.sum())
    )

    print(
        "Testing positives:",
        int(y_test.sum())
    )

    model = RandomForestClassifier(

        n_estimators=300,

        max_depth=8,

        min_samples_leaf=2,

        class_weight="balanced",

        random_state=42,

        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    scores = model.predict_proba(
        X_test
    )[:, 1]

    test = data.iloc[
        test_idx
    ].copy()

    test["ml_score"] = scores

    print("\nCLASSIFICATION METRICS")

    print(
        "ROC AUC:",
        round(
            roc_auc_score(
                y_test, scores
            ),
            4
        )
    )

    print(
        "Average precision:",
        round(
            average_precision_score(
                y_test, scores
            ),
            4
        )
    )

    # Compare both ranking methods on
    # the exact same held-out candidate pool.

    physics = evaluate_ranking(
        test,
        "physics"
    )

    ml = evaluate_ranking(
        test,
        "ml"
    )

    ranking = pd.concat(

        [physics, ml],

        ignore_index=True
    )

    print("\nRANKING PERFORMANCE")

    for method, group in ranking.groupby(
        "method"
    ):

        print(
            "\nMethod:",
            method
        )

        print(
            "Candidate pool recall:",
            round(
                group[
                    "candidate_available"
                ].mean() * 100,
                2
            ),
            "%"
        )

        print(
            "Top-1 recovery:",
            round(
                group["top1"].mean() * 100,
                2
            ),
            "%"
        )

        print(
            "Top-5 recovery:",
            round(
                group["top5"].mean() * 100,
                2
            ),
            "%"
        )

        print(
            "Top-10 recovery:",
            round(
                group["top10"].mean() * 100,
                2
            ),
            "%"
        )

    importance = pd.DataFrame({

        "feature": FEATURES,

        "importance":
            model.feature_importances_

    }).sort_values(

        "importance",

        ascending=False
    )

    print("\nFEATURE IMPORTANCE")

    print(

        importance.head(10)

        .to_string(index=False)

    )

    joblib.dump(

        {

            "model": model,

            "features": FEATURES

        },

        MODEL_DIR / "planet_ranker.joblib"
    )

    test.to_csv(

        DATA_DIR / "test_predictions.csv",

        index=False
    )

    ranking.to_csv(

        DATA_DIR / "ranking_metrics.csv",

        index=False
    )

    importance.to_csv(

        DATA_DIR / "feature_importance.csv",

        index=False
    )

    print("\nMODEL SAVED")

    print(
        MODEL_DIR / "planet_ranker.joblib"
    )


if __name__ == "__main__":
    main()
