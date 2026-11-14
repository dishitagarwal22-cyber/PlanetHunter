
import numpy as np
import pandas as pd

from itertools import combinations

from astropy.coordinates import SkyCoord
import astropy.units as u

from motion_models import (
    design_matrices,
    fit_batch,
    REFERENCE_RA,
    REFERENCE_DEC
)


CATALOGUE = "data/source_catalogue.csv"

OUTPUT = "data/moving_candidates.csv"

MIN_DISTANCE = 200
MAX_DISTANCE = 1500

TOP_PER_EPOCH_SET = 100

# Preliminary screening thresholds.
# These are not calibrated uncertainties.

MAX_PARALLAX_RMS = 3.0
MIN_STATIONARY_RMS = 3.0
MAX_SPEED_RATIO = 2.0


def evaluate_epoch_set(
    catalogue,
    epoch_ids
):

    groups = [
        catalogue[
            catalogue["epoch_id"] == epoch
        ].reset_index(drop=True)

        for epoch in epoch_ids
    ]

    sizes = [len(group) for group in groups]

    if min(sizes) == 0:
        return pd.DataFrame(), 0

    # Generate all possible combinations
    # of one detection per selected epoch.

    indices = np.indices(
        sizes
    ).reshape(len(groups), -1).T

    n_tracks = len(indices)

    n_epochs = len(groups)

    observations = np.empty(
        (n_tracks, 2 * n_epochs)
    )

    source_ids = []

    snrs = np.empty(
        (n_tracks, n_epochs)
    )

    mjd = []

    for k, group in enumerate(groups):

        selected = group.iloc[
            indices[:, k]
        ]

        observations[:, 2*k] = (
            selected["east_arcsec"].to_numpy()
        )

        observations[:, 2*k + 1] = (
            selected["north_arcsec"].to_numpy()
        )

        source_ids.append(
            selected["source_id"].to_numpy()
        )

        snrs[:, k] = (
            selected["peak_snr"].to_numpy()
        )

        mjd.append(
            float(group["mjd"].iloc[0])
        )

    matrices = design_matrices(mjd)

    results = fit_batch(
        observations,
        matrices,
        min_distance=MIN_DISTANCE,
        max_distance=MAX_DISTANCE
    )

    df = pd.DataFrame({

        "n_epochs": n_epochs,

        "epoch_ids": ";".join(
            map(str, epoch_ids)
        ),

        "stationary_rms":
            results["stationary_rms"],

        "linear_rms":
            results["linear_rms"],

        "parallax_rms":
            results["parallax_rms"],

        "distance_au":
            results["distance_au"],

        "raw_parallax":
            results["raw_parallax"],

        "parallax_arcsec":
            results["parallax_arcsec"],

        "vx_arcsec_year":
            results["vx_arcsec_year"],

        "vy_arcsec_year":
            results["vy_arcsec_year"],

        "speed_arcsec_year":
            results["speed_arcsec_year"],

        "circular_speed_ratio":
            results["circular_speed_ratio"],

        "parallax_at_boundary":
            results["parallax_at_boundary"],

        "mean_peak_snr":
            np.mean(snrs, axis=1),

        "min_peak_snr":
            np.min(snrs, axis=1),

        "parallax_leverage":
            results["parallax_leverage"]
    })

    # Store individual source IDs.

    for k, epoch in enumerate(epoch_ids):

        df[f"source_E{epoch}"] = (
            source_ids[k]
        )

    # How much does the parallax model
    # improve upon the alternatives?

    df["improvement_over_stationary"] = (
        df["stationary_rms"]
        - df["parallax_rms"]
    )

    df["improvement_over_linear"] = (
        df["linear_rms"]
        - df["parallax_rms"]
    )

    # Preliminary candidate screening.
    # Keep flags rather than permanently
    # deleting candidates.

    df["passes_screen"] = (

        (df["parallax_rms"]
         <= MAX_PARALLAX_RMS)

        &

        (df["stationary_rms"]
         >= MIN_STATIONARY_RMS)

        &

        (df["circular_speed_ratio"]
         <= MAX_SPEED_RATIO)

        &

        (~df["parallax_at_boundary"])
    )

    # Physics-only baseline ranking.
    # This is a heuristic, not a probability.

    df["physics_score"] = (

        df["parallax_rms"]

        + 2.0 * np.maximum(
            df["circular_speed_ratio"] - 1.5,
            0
        )

        + 3.0 * df[
            "parallax_at_boundary"
        ].astype(float)

        + np.maximum(
            MIN_STATIONARY_RMS
            - df["stationary_rms"],
            0
        )
    )

    # Lower score is better.

    df = df.sort_values(
        ["passes_screen", "physics_score"],
        ascending=[False, True]
    )

    print(
        f"\nEpochs {epoch_ids}"
    )

    print(
        f"Combinations evaluated: {n_tracks}"
    )

    print(
        "Passing preliminary screen:",
        int(df["passes_screen"].sum())
    )

    print(
        "Best parallax RMS:",
        round(df["parallax_rms"].min(), 3),
        "arcsec"
    )

    return (
        df.head(TOP_PER_EPOCH_SET).copy(),
        n_tracks
    )


def main():

    catalogue = pd.read_csv(CATALOGUE)

    # Convert RA/Dec into a common
    # local tangent-plane coordinate system.

    reference = SkyCoord(
        ra=REFERENCE_RA * u.deg,
        dec=REFERENCE_DEC * u.deg
    )

    coords = SkyCoord(
        ra=catalogue["ra"].to_numpy() * u.deg,
        dec=catalogue["dec"].to_numpy() * u.deg
    )

    east, north = (
        reference.spherical_offsets_to(coords)
    )

    catalogue["east_arcsec"] = east.arcsec

    catalogue["north_arcsec"] = north.arcsec

    epochs = sorted(
        catalogue["epoch_id"].unique()
    )

    all_candidates = []

    total_evaluated = 0

    # Four-epoch tracks first,
    # then three-epoch tracks.

    for n in [4, 3]:

        if len(epochs) < n:
            continue

        for epoch_set in combinations(epochs, n):

            candidates, count = (
                evaluate_epoch_set(
                    catalogue,
                    epoch_set
                )
            )

            total_evaluated += count

            if not candidates.empty:

                all_candidates.append(
                    candidates
                )

    if not all_candidates:

        raise ValueError(
            "No candidate tracks generated"
        )

    final = pd.concat(
        all_candidates,
        ignore_index=True
    )

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

    final.insert(
        0,
        "candidate_id",
        [
            f"M{i:04d}"
            for i in range(
                1, len(final) + 1
            )
        ]
    )

    final.to_csv(
        OUTPUT,
        index=False
    )

    print("\nTRAJECTORY SEARCH SUMMARY")

    print(
        "Total combinations evaluated:",
        total_evaluated
    )

    print(
        "Candidates retained:",
        len(final)
    )

    print(
        "Retained candidates passing screen:",
        int(final["passes_screen"].sum())
    )

    columns = [
        "candidate_id",
        "n_epochs",
        "epoch_ids",
        "parallax_rms",
        "linear_rms",
        "stationary_rms",
        "distance_au",
        "circular_speed_ratio",
        "passes_screen"
    ]

    print("\nTOP CANDIDATES")

    print(
        final[columns]
        .head(15)
        .round(3)
        .to_string(index=False)
    )

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
