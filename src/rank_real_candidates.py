
import numpy as np
import pandas as pd
import joblib

from pathlib import Path


CANDIDATE_FILE = Path(
    "data/moving_candidates.csv"
)

MODEL_FILE = Path(
    "models/planet_ranker.joblib"
)

OUTPUT_FILE = Path(
    "data/ml_ranked_candidates.csv"
)


def main():

    # Load the trained Random Forest.

    saved = joblib.load(MODEL_FILE)

    model = saved["model"]

    features = saved["features"]

    # Load real SPHEREx candidate tracks.

    candidates = pd.read_csv(
        CANDIDATE_FILE
    )

    print(
        "Real candidates loaded:",
        len(candidates)
    )

    # Preserve the original physics ranking.

    candidates = candidates.sort_values(

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

    ).reset_index(drop=True)

    candidates["physics_rank"] = (
        np.arange(
            1, len(candidates) + 1
        )
    )

    # Extract the same features used
    # during model training.

    X = candidates[features].astype(float)

    X = X.replace(

        [np.inf, -np.inf],

        np.nan
    )

    valid = X.notna().all(axis=1)

    # Missing or invalid features are
    # left unscored rather than invented.

    candidates["ml_score"] = np.nan

    if valid.any():

        candidates.loc[
            valid, "ml_score"
        ] = model.predict_proba(

            X.loc[valid]

        )[:, 1]

    # Rank using the ML score.
    # Physics rank breaks exact ties.

    candidates = candidates.sort_values(

        [
            "ml_score",
            "physics_rank"
        ],

        ascending=[
            False,
            True
        ],

        na_position="last"

    ).reset_index(drop=True)

    candidates["ml_rank"] = np.arange(
        1, len(candidates) + 1
    )

    candidates["rank_change"] = (

        candidates["physics_rank"]

        -

        candidates["ml_rank"]
    )

    # Save complete ranked catalogue.

    candidates.to_csv(

        OUTPUT_FILE,

        index=False
    )

    columns = [

        "candidate_id",

        "epoch_ids",

        "physics_rank",

        "ml_rank",

        "ml_score",

        "distance_au",

        "parallax_rms",

        "circular_speed_ratio",

        "passes_screen"

    ]

    print("\nTOP 15 ML-RANKED CANDIDATES")

    print(

        candidates[columns]

        .head(15)

        .round(4)

        .to_string(index=False)
    )

    print(
        "\nPHYSICALLY SCREENED CANDIDATES"
    )

    screened = candidates[

        candidates["passes_screen"]

        == True

    ]

    print(

        screened[columns]

        .round(4)

        .to_string(index=False)
    )

    print(
        "\nHighest-scoring candidate:"
    )

    best = candidates.iloc[0]

    print(
        "Candidate ID:",
        best["candidate_id"]
    )

    print(
        "ML score:",
        round(
            best["ml_score"], 4
        )
    )

    print(
        "Estimated distance:",
        round(
            best["distance_au"], 2
        ),
        "AU"
    )

    print(
        "Physics rank:",
        best["physics_rank"]
    )

    print(
        "Passes physical screen:",
        best["passes_screen"]
    )

    print(
        "\nSaved ranked catalogue:",
        OUTPUT_FILE
    )

    print(
        "\nIMPORTANT: ML scores are "
        "candidate ranking scores, "
        "not planet discovery probabilities."
    )


if __name__ == "__main__":
    main()
