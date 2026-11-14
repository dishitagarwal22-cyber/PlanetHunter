
import numpy as np
import pandas as pd

from itertools import combinations
from pathlib import Path

import trajectory_linking as linking


DATA_DIR = Path("data/synthetic")


def main():

    catalogue = pd.read_csv(
        DATA_DIR / "source_catalogue.csv"
    )

    truth = pd.read_csv(
        DATA_DIR / "ground_truth.csv"
    )

    epochs = sorted(
        catalogue["epoch_id"].unique()
    )

    recovered_truth = truth[
        truth["recovered"] == True
    ]

    recovered_ids = dict(zip(

        recovered_truth["epoch_id"],

        recovered_truth["recovered_source_id"]

    ))

    print("GROUND TRUTH")

    print(
        truth[[
            "epoch_id",
            "recovered",
            "recovered_source_id",
            "centroid_error_pixels"
        ]].to_string(index=False)
    )

    # Keep every evaluated candidate for
    # this diagnostic, so that we can measure
    # the exact rank of the true association.

    linking.TOP_PER_EPOCH_SET = 10**9

    all_candidates = []

    total_evaluated = 0

    # Evaluate all three-epoch and
    # four-epoch combinations.

    for n in [4, 3]:

        for epoch_set in combinations(
            epochs, n
        ):

            candidates, count = (
                linking.evaluate_epoch_set(
                    catalogue,
                    epoch_set
                )
            )

            total_evaluated += count

            if candidates.empty:
                continue

            candidates = (
                candidates.reset_index(drop=True)
            )

            candidates["rank_in_subset"] = (
                np.arange(
                    1, len(candidates) + 1
                )
            )

            # A correct track must contain
            # the injected object's recovered
            # detection in every selected epoch.

            correct = np.ones(
                len(candidates),
                dtype=bool
            )

            for epoch in epoch_set:

                if epoch not in recovered_ids:

                    correct[:] = False
                    break

                correct &= (

                    candidates[f"source_E{epoch}"]

                    == recovered_ids[epoch]

                ).to_numpy()

            candidates["is_true_track"] = correct

            all_candidates.append(
                candidates
            )

    final = pd.concat(

        all_candidates,

        ignore_index=True

    )

    # Reproduce the ranking convention
    # used by our real-data search.

    final = final.sort_values(

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

    final["global_rank"] = np.arange(
        1, len(final) + 1
    )

    true_tracks = final[
        final["is_true_track"]
    ]

    print("\nFINAL RECOVERY SUMMARY")

    print(
        "Total detections recovered:",
        len(recovered_truth),
        "/",
        len(truth)
    )

    print(
        "Total trajectories evaluated:",
        total_evaluated
    )

    print(
        "Correct associations:",
        len(true_tracks)
    )

    print(
        "Candidates passing screen:",
        int(final["passes_screen"].sum())
    )

    if true_tracks.empty:

        print(
            "No correct trajectory "
            "could be constructed."
        )

    else:

        for _, row in true_tracks.iterrows():

            true_distance = float(
                truth["true_distance_au"].iloc[0]
            )

            fitted_distance = float(
                row["distance_au"]
            )

            distance_error = (

                abs(
                    fitted_distance - true_distance
                )

                / true_distance * 100

            )

            print("\nCORRECT TRACK FOUND")

            print(
                "Epochs:",
                row["epoch_ids"]
            )

            print(
                "Rank within epoch subset:",
                int(row["rank_in_subset"])
            )

            print(
                "Global rank:",
                int(row["global_rank"])
            )

            print(
                "True distance:",
                round(true_distance, 3),
                "AU"
            )

            print(
                "Fitted distance:",
                round(fitted_distance, 3),
                "AU"
            )

            print(
                "Distance error:",
                round(distance_error, 3),
                "%"
            )

            print(
                "Parallax RMS:",
                round(row["parallax_rms"], 3),
                "arcsec"
            )

            print(
                "Linear RMS:",
                round(row["linear_rms"], 3),
                "arcsec"
            )

            print(
                "Stationary RMS:",
                round(row["stationary_rms"], 3),
                "arcsec"
            )

            print(
                "Speed ratio:",
                round(
                    row["circular_speed_ratio"], 3
                )
            )

            print(
                "Distance at boundary:",
                row["parallax_at_boundary"]
            )

            print(
                "Passes screen:",
                row["passes_screen"]
            )

    # Save a compact diagnostic catalogue:
    # top 1000 candidates plus every true track.

    retained = pd.concat([

        final.head(1000),

        true_tracks

    ]).drop_duplicates(
        subset="global_rank"
    )

    retained.to_csv(

        DATA_DIR / "evaluated_candidates.csv",

        index=False

    )

    true_tracks.to_csv(

        DATA_DIR / "true_tracks.csv",

        index=False

    )

    print(
        "\nSaved recovery evaluation."
    )


if __name__ == "__main__":
    main()
