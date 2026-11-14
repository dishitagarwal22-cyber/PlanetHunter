
import numpy as np
import pandas as pd

from pathlib import Path
from itertools import combinations

from astropy.coordinates import SkyCoord
import astropy.units as u

import synthetic_recovery as synth
import trajectory_linking as linking

from source_detection import detect_sources


OUTPUT = Path("data/ml")
OUTPUT.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(2026)

N_SIMULATIONS = 40

# Retain the top 100 candidates from
# each epoch combination.
linking.TOP_PER_EPOCH_SET = 100

# Limit expensive trajectory searches.
synth.N_ATTEMPTS = 5000

REFERENCE = synth.REFERENCE


def make_catalogue(
    observations,
    injection_positions,
    peak_snr,
    sigma
):

    records = []
    recovered_ids = {}

    synth.PEAK_SNR = peak_snr
    synth.SIGMA = sigma

    for obs in observations:

        epoch = obs["epoch_id"]

        image = obs["image"]

        injected = (
            epoch in injection_positions
        )

        if injected:

            x_true, y_true = (
                injection_positions[epoch]
            )

            image = synth.inject_gaussian(
                image,
                x_true,
                y_true
            )

            sources = detect_sources(image)

        else:

            sources = obs["original_sources"]

        if injected and len(sources) > 0:

            distances = np.array([

                np.hypot(
                    s["x"] - x_true,
                    s["y"] - y_true
                )

                for s in sources
            ])

            closest = int(
                np.argmin(distances)
            )

            if distances[closest] <= 1.0:

                recovered_ids[epoch] = (
                    f"E{epoch}_{closest+1:03d}"
                )

        for j, source in enumerate(
            sources, 1
        ):

            source_id = (
                f"E{epoch}_{j:03d}"
            )

            ra, dec = (
                obs["wcs"].pixel_to_world_values(
                    source["x"],
                    source["y"]
                )
            )

            coord = SkyCoord(

                ra=float(ra) * u.deg,

                dec=float(dec) * u.deg
            )

            east, north = (
                REFERENCE.spherical_offsets_to(
                    coord
                )
            )

            records.append({

                "epoch_id": epoch,

                "source_id": source_id,

                "mjd": obs["mjd"],

                "east_arcsec": east.arcsec,

                "north_arcsec": north.arcsec,

                "peak_snr":
                    source["peak_snr"]

            })

    return (
        pd.DataFrame(records),
        recovered_ids
    )


def generate_candidates(
    catalogue,
    recovered_ids,
    simulation_id
):

    epochs = sorted(
        catalogue["epoch_id"].unique()
    )

    results = []

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

            if candidates.empty:
                continue

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

            candidates["label"] = (
                correct.astype(int)
            )

            candidates["simulation_id"] = (
                simulation_id
            )

            results.append(candidates)

    return pd.concat(
        results,
        ignore_index=True
    )


def main():

    observations = synth.load_epochs()

    all_training_rows = []
    summaries = []

    epoch_options = list(
        combinations(
            range(len(observations)), 3
        )
    )

    epoch_options.append(
        tuple(range(len(observations)))
    )

    for simulation_id in range(
        1, N_SIMULATIONS + 1
    ):

        print(
            "\n" + "=" * 50
        )

        print(
            f"SIMULATION {simulation_id}"
            f"/{N_SIMULATIONS}"
        )

        # Choose three or four epochs.
        # Every simulation still processes
        # all four real background images.

        option_index = int(
            RNG.integers(
                len(epoch_options)
            )
        )

        selected_indices = (
            epoch_options[option_index]
        )

        selected_observations = [

            observations[i]

            for i in selected_indices
        ]

        try:

            trajectory = (
                synth.find_valid_trajectory(
                    selected_observations
                )
            )

        except RuntimeError:

            print(
                "No valid trajectory found."
            )

            summaries.append({

                "simulation_id":
                    simulation_id,

                "status":
                    "no_valid_trajectory"

            })

            continue

        injection_positions = {

            obs["epoch_id"]: position

            for obs, position in zip(

                selected_observations,

                trajectory["positions"]

            )
        }

        # Vary source brightness and width.

        peak_snr = float(
            RNG.uniform(6, 30)
        )

        sigma = float(
            RNG.uniform(0.8, 1.5)
        )

        catalogue, recovered_ids = (
            make_catalogue(

                observations,

                injection_positions,

                peak_snr,

                sigma
            )
        )

        candidates = generate_candidates(

            catalogue,

            recovered_ids,

            simulation_id
        )

        positive_count = int(
            candidates["label"].sum()
        )

        n_injected = len(
            injection_positions
        )

        n_recovered = len(
            recovered_ids
        )

        summaries.append({

            "simulation_id":
                simulation_id,

            "status": "completed",

            "true_distance_au":
                trajectory["distance_au"],

            "peak_snr": peak_snr,

            "sigma": sigma,

            "n_injected": n_injected,

            "n_recovered": n_recovered,

            "positive_tracks":
                positive_count,

            "candidate_count":
                len(candidates),

            "trajectory_attempts":
                trajectory["attempts"]

        })

        all_training_rows.append(
            candidates
        )

        print(
            "Distance:",
            round(
                trajectory["distance_au"], 2
            ),
            "AU"
        )

        print(
            "Recovered:",
            n_recovered,
            "/",
            n_injected
        )

        print(
            "Correct tracks in candidate pool:",
            positive_count
        )

        # Checkpoint after each simulation.
        # Results survive interruptions.

        pd.concat(

            all_training_rows,

            ignore_index=True

        ).to_csv(

            OUTPUT / "training_data.csv",

            index=False
        )

        pd.DataFrame(summaries).to_csv(

            OUTPUT / "simulation_summary.csv",

            index=False
        )

    if not all_training_rows:

        raise RuntimeError(
            "No successful simulations"
        )

    training = pd.concat(

        all_training_rows,

        ignore_index=True
    )

    summary = pd.DataFrame(summaries)

    training.to_csv(

        OUTPUT / "training_data.csv",

        index=False
    )

    summary.to_csv(

        OUTPUT / "simulation_summary.csv",

        index=False
    )

    completed = summary[
        summary["status"] == "completed"
    ]

    print("\nFINAL DATASET SUMMARY")

    print(
        "Completed simulations:",
        len(completed)
    )

    print(
        "Training rows:",
        len(training)
    )

    print(
        "Positive tracks:",
        int(training["label"].sum())
    )

    print(
        "Negative tracks:",
        int((training["label"] == 0).sum())
    )

    print(
        "Simulations with correct track:",
        int(
            (
                completed["positive_tracks"] > 0
            ).sum()
        )
    )

    detection_recall = (

        completed["n_recovered"].sum()

        /

        completed["n_injected"].sum()
    )

    print(
        "Detection recall:",
        round(
            detection_recall * 100, 2
        ),
        "%"
    )

    print(
        "Saved dataset in:",
        OUTPUT
    )


if __name__ == "__main__":
    main()
