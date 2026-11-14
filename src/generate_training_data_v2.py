
from pathlib import Path

import numpy as np
import pandas as pd

import astropy.units as u

import synthetic_recovery as synth
import trajectory_linking as linking

from generate_training_data import (
    make_catalogue,
    generate_candidates
)

from motion_models import (
    parallax_factors,
    ARCSEC_PER_RADIAN
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

OUTPUT = Path("data/ml_v2")

OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)

RNG = np.random.default_rng(2027)

DISTANCE_BINS = [
    (200, 400),
    (400, 600),
    (600, 800),
    (800, 1000),
    (1000, 1200),
    (1200, 1500)
]

# Start with 3 per bin for testing.
# Increase to 20 after the test succeeds.

TARGET_PER_BIN = 3

MAX_PROPOSALS_PER_BIN = 5000

BORDER = 5

linking.TOP_PER_EPOCH_SET = 100

REFERENCE = synth.REFERENCE


# --------------------------------------------------
# GENERATE A PHYSICAL TRAJECTORY
# --------------------------------------------------

def propose_trajectory(
    observations,
    low,
    high,
    t,
    px,
    py
):

    distance = RNG.uniform(
        low, high
    )

    parallax = (
        ARCSEC_PER_RADIAN / distance
    )

    circular_speed = (
        1296000 / distance**1.5
    )

    speed = circular_speed * RNG.uniform(
        0.2, 1.2
    )

    angle = RNG.uniform(
        0, 2*np.pi
    )

    vx = speed * np.cos(angle)

    vy = speed * np.sin(angle)

    east0 = RNG.uniform(
        -150, 150
    )

    north0 = RNG.uniform(
        -150, 150
    )

    east = (

        east0

        + vx*t

        + parallax*(px-px[0])
    )

    north = (

        north0

        + vy*t

        + parallax*(py-py[0])
    )

    positions = {}

    epoch_details = []

    # Calculate positions for ALL epochs.

    for j, obs in enumerate(
        observations
    ):

        epoch_id = obs["epoch_id"]

        coord = (
            REFERENCE.spherical_offsets_by(

                east[j]*u.arcsec,

                north[j]*u.arcsec
            )
        )

        x, y = obs["wcs"].world_to_pixel(
            coord
        )

        x = float(x)
        y = float(y)

        image = obs["image"]

        height, width = image.shape

        status = "visible"

        if not (
            np.isfinite(x)
            and np.isfinite(y)
        ):

            status = "invalid_coordinate"

        elif not (
            0 <= x < width
            and 0 <= y < height
        ):

            status = "outside_field"

        elif not (
            BORDER <= x < width-BORDER
            and BORDER <= y < height-BORDER
        ):

            status = "edge_exclusion"

        else:

            xi = int(round(x))
            yi = int(round(y))

            patch = image[
                yi-3:yi+4,
                xi-3:xi+4
            ]

            if (
                patch.shape != (7, 7)
                or not np.isfinite(
                    patch
                ).all()
            ):

                status = "invalid_pixels"

        if status == "visible":

            positions[epoch_id] = (
                x, y
            )

        epoch_details.append({

            "epoch_id": epoch_id,

            "true_x": x,

            "true_y": y,

            "east_arcsec": east[j],

            "north_arcsec": north[j],

            "visibility_status": status
        })

    return {

        "distance_au": distance,

        "parallax_arcsec": parallax,

        "vx_arcsec_year": vx,

        "vy_arcsec_year": vy,

        "positions": positions,

        "epoch_details": epoch_details
    }


# --------------------------------------------------
# CHECKPOINT RESULTS
# --------------------------------------------------

def save_results(
    datasets,
    summaries,
    proposals,
    epoch_records
):

    if datasets:

        pd.concat(
            datasets,
            ignore_index=True
        ).to_csv(

            OUTPUT / "training_data.csv",

            index=False
        )

    pd.DataFrame(
        summaries
    ).to_csv(

        OUTPUT / "simulation_summary.csv",

        index=False
    )

    pd.DataFrame(
        proposals
    ).to_csv(

        OUTPUT / "geometric_proposals.csv",

        index=False
    )

    pd.DataFrame(
        epoch_records
    ).to_csv(

        OUTPUT / "epoch_truth.csv",

        index=False
    )


# --------------------------------------------------
# MAIN SIMULATION LOOP
# --------------------------------------------------

def main():

    observations = synth.load_epochs()

    mjd = np.array([

        obs["mjd"]

        for obs in observations
    ])

    t = (
        mjd - mjd[0]
    ) / 365.25

    px, py = parallax_factors(
        mjd
    )

    datasets = []

    summaries = []

    proposals = []

    epoch_records = []

    simulation_id = 0

    proposal_id = 0

    for low, high in DISTANCE_BINS:

        accepted = 0

        attempts = 0

        print(
            f"\nDISTANCE BIN: {low}-{high} AU"
        )

        while (
            accepted < TARGET_PER_BIN
            and attempts < MAX_PROPOSALS_PER_BIN
        ):

            attempts += 1

            proposal_id += 1

            trajectory = propose_trajectory(

                observations,

                low,

                high,

                t,

                px,

                py
            )

            positions = trajectory[
                "positions"
            ]

            n_visible = len(
                positions
            )

            proposals.append({

                "proposal_id": proposal_id,

                "distance_bin": f"{low}-{high}",

                "true_distance_au":
                    trajectory["distance_au"],

                "n_visible": n_visible,

                "geometrically_usable":
                    n_visible >= 3
            })

            # Preserve rejected geometric
            # proposals instead of pretending
            # every object is observable.

            if n_visible < 3:

                continue

            simulation_id += 1

            accepted += 1

            # Randomize brightness and width.

            peak_snr = RNG.uniform(
                6, 30
            )

            sigma = RNG.uniform(
                0.8, 1.5
            )

            print(
                f"\nSimulation {simulation_id}"
            )

            print(

                "Distance:",

                round(
                    trajectory["distance_au"],
                    2
                ),

                "AU"
            )

            print(

                "Visible epochs:",

                list(positions.keys())
            )

            # Existing injection and
            # source-detection pipeline.

            catalogue, recovered_ids = (

                make_catalogue(

                    observations,

                    positions,

                    peak_snr,

                    sigma
                )
            )

            # Existing physics-based
            # candidate generation.

            candidates = generate_candidates(

                catalogue,

                recovered_ids,

                simulation_id
            )

            candidates["distance_bin"] = (
                f"{low}-{high}"
            )

            candidates["true_distance_au"] = (
                trajectory["distance_au"]
            )

            candidates["n_visible_epochs"] = (
                n_visible
            )

            candidates["injected_peak_snr"] = (
                peak_snr
            )

            datasets.append(
                candidates
            )

            n_positive = int(
                candidates["label"].sum()
            )

            n_recovered = len(
                recovered_ids
            )

            summaries.append({

                "simulation_id": simulation_id,

                "proposal_id": proposal_id,

                "status": "completed",

                "distance_bin": f"{low}-{high}",

                "true_distance_au":
                    trajectory["distance_au"],

                "true_vx":
                    trajectory["vx_arcsec_year"],

                "true_vy":
                    trajectory["vy_arcsec_year"],

                "peak_snr": peak_snr,

                "sigma": sigma,

                "n_injected": n_visible,

                "n_recovered": n_recovered,

                "positive_tracks": n_positive,

                "candidate_count":
                    len(candidates),

                "trajectory_attempts": attempts
            })

            # Record why an object was
            # missing in every epoch.

            for detail in trajectory[
                "epoch_details"
            ]:

                epoch_id = detail[
                    "epoch_id"
                ]

                recovered = (
                    epoch_id in recovered_ids
                )

                epoch_records.append({

                    "simulation_id":
                        simulation_id,

                    **detail,

                    "injected":
                        epoch_id in positions,

                    "recovered":
                        recovered,

                    "recovered_source_id":
                        recovered_ids.get(
                            epoch_id
                        )
                })

            print(

                "Recovered detections:",

                n_recovered,

                "/",

                n_visible
            )

            print(

                "Correct retained tracks:",

                n_positive
            )

            # Save after every completed
            # simulation.

            save_results(

                datasets,

                summaries,

                proposals,

                epoch_records
            )

        print(

            f"\nCompleted bin {low}-{high} AU:",

            accepted,

            "/",

            TARGET_PER_BIN
        )

    save_results(

        datasets,

        summaries,

        proposals,

        epoch_records
    )

    summary = pd.DataFrame(
        summaries
    )

    if summary.empty:

        print(
            "No usable simulations generated."
        )

        return

    data = pd.concat(

        datasets,

        ignore_index=True
    )

    print("\nFINAL V2 DATASET SUMMARY")

    print(

        "Completed simulations:",

        len(summary)
    )

    print(

        "Training rows:",

        len(data)
    )

    print(

        "Positive tracks:",

        int(data["label"].sum())
    )

    print(

        "Negative tracks:",

        int((data["label"] == 0).sum())
    )

    print(

        "Detection recall:",

        round(

            100

            * summary["n_recovered"].sum()

            / summary["n_injected"].sum(),

            2
        ),

        "%"
    )

    print(

        "Simulations with correct track:",

        int(

            (
                summary["positive_tracks"] > 0
            ).sum()
        )
    )

    print("\nRESULTS BY DISTANCE BIN")

    for distance_bin, group in summary.groupby(
        "distance_bin",
        sort=False
    ):

        detection_recall = (

            100

            * group["n_recovered"].sum()

            / group["n_injected"].sum()
        )

        track_recall = (

            100

            * (
                group["positive_tracks"] > 0
            ).mean()
        )

        print(

            distance_bin,

            "| Simulations:", len(group),

            "| Detection recall:",
                round(detection_recall, 2),

            "| Candidate pool recall:",
                round(track_recall, 2)
        )

    print("\nDISTANCE DISTRIBUTION")

    print(
        summary["true_distance_au"].describe()
    )

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
