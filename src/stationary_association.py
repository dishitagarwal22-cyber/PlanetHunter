
import numpy as np
import pandas as pd

from itertools import combinations
from astropy.coordinates import SkyCoord
import astropy.units as u


CATALOGUE = "data/source_catalogue.csv"

OUTPUT = "data/stationary_pairs.csv"

MAX_SEPARATION = 6.1  # arcseconds


def main():

    catalogue = pd.read_csv(CATALOGUE)

    epochs = sorted(
        catalogue["epoch_id"].unique()
    )

    records = []

    print("STATIONARY ASSOCIATION")
    print("Total detections:", len(catalogue))

    for epoch1, epoch2 in combinations(epochs, 2):

        sources1 = catalogue[
            catalogue["epoch_id"] == epoch1
        ].reset_index(drop=True)

        sources2 = catalogue[
            catalogue["epoch_id"] == epoch2
        ].reset_index(drop=True)

        coords1 = SkyCoord(
            ra=sources1["ra"].to_numpy() * u.deg,
            dec=sources1["dec"].to_numpy() * u.deg
        )

        coords2 = SkyCoord(
            ra=sources2["ra"].to_numpy() * u.deg,
            dec=sources2["dec"].to_numpy() * u.deg
        )

        # Calculate every pairwise separation

        separations = coords1[:, None].separation(
            coords2[None, :]
        ).arcsec

        rows, cols = np.where(
            separations <= MAX_SEPARATION
        )

        print(
            f"\nEpoch {epoch1} vs Epoch {epoch2}"
        )

        print(
            f"Possible stationary pairs: {len(rows)}"
        )

        for i, j in zip(rows, cols):

            a = sources1.iloc[i]
            b = sources2.iloc[j]

            separation = separations[i, j]

            pixel_distance = np.hypot(
                a["x"] - b["x"],
                a["y"] - b["y"]
            )

            dt_days = abs(
                a["mjd"] - b["mjd"]
            )

            # Is this the closest detection
            # in both directions?

            mutual_nearest = (
                j == np.argmin(separations[i, :])
                and
                i == np.argmin(separations[:, j])
            )

            records.append({
                "epoch1": epoch1,
                "epoch2": epoch2,
                "source1": a["source_id"],
                "source2": b["source_id"],
                "separation_arcsec": separation,
                "pixel_distance": pixel_distance,
                "dt_days": dt_days,
                "mutual_nearest": mutual_nearest
            })

            print(
                f"{a['source_id']} <-> "
                f"{b['source_id']} | "
                f"{separation:.2f} arcsec | "
                f"Mutual nearest: {mutual_nearest}"
            )

    columns = [
        "epoch1",
        "epoch2",
        "source1",
        "source2",
        "separation_arcsec",
        "pixel_distance",
        "dt_days",
        "mutual_nearest"
    ]

    results = pd.DataFrame(
        records,
        columns=columns
    )

    results.to_csv(
        OUTPUT,
        index=False
    )

    print("\nSUMMARY")

    print(
        "Total possible stationary pairs:",
        len(results)
    )

    if not results.empty:

        mutual = results[
            results["mutual_nearest"]
        ]

        print(
            "Mutual nearest pairs:",
            len(mutual)
        )

        print(
            "Median separation:",
            round(
                mutual["separation_arcsec"].median(),
                3
            ),
            "arcsec"
        )

    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
