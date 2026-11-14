
import pandas as pd

from synthetic_recovery import load_epochs

from candidate_features import (
    augment_candidates,
    NEW_FEATURES
)


def main():

    catalogue = pd.read_csv(

        "data/source_catalogue.csv"
    )

    candidates = pd.read_csv(

        "data/ml_ranked_candidates.csv"
    )

    observations = load_epochs()

    enriched = augment_candidates(

        candidates,

        catalogue,

        observations
    )

    output = (

        "data/enriched_real_candidates.csv"
    )

    enriched.to_csv(

        output,

        index=False
    )

    print("\nPHYSICS TOP 10")

    columns = [

        "candidate_id",

        "distance_au",

        "parallax_rms",

        "stationary_member_fraction",

        "mean_stationary_support",

        "max_fit_residual",

        "escape_speed_ratio",

        "missing_inside_field",

        "missing_outside_field"
    ]

    print(

        enriched.sort_values(

            "physics_rank"

        )[columns]

        .head(10)

        .round(3)

        .to_string(index=False)
    )

    print("\nML TOP 5")

    print(

        enriched.sort_values(

            "ml_rank"

        )[columns]

        .head(5)

        .round(3)

        .to_string(index=False)
    )

    print("\nMISSING FEATURE VALUES")

    print(

        enriched[NEW_FEATURES]

        .isna()

        .sum()
    )

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
