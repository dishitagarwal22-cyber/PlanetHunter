
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

from source_detection import detect_sources

from motion_models import (
    parallax_factors,
    REFERENCE_RA,
    REFERENCE_DEC,
    ARCSEC_PER_RADIAN
)

from trajectory_linking import evaluate_epoch_set


# -----------------------------
# Configuration
# -----------------------------

OUTPUT_DIR = Path("data/synthetic")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RNG = np.random.default_rng(42)

MIN_DISTANCE = 200
MAX_DISTANCE = 1500

N_ATTEMPTS = 20000

# Gaussian source parameters

SIGMA = 1.1
PEAK_SNR = 25.0

BORDER = 5

# Require the injection to be away
# from previously detected sources.

MIN_SOURCE_DISTANCE = 4.0

REFERENCE = SkyCoord(
    ra=REFERENCE_RA * u.deg,
    dec=REFERENCE_DEC * u.deg
)


def load_epochs():

    epochs = pd.read_csv(
        "data/epochs.csv"
    ).sort_values("mjd")

    observations = []

    for epoch_id, (_, row) in enumerate(
        epochs.iterrows(), 1
    ):

        with fits.open(
            row["aligned"]
        ) as hdul:

            image = hdul[0].data.astype(float)

            header = hdul[0].header.copy()

            wcs = WCS(header).celestial

        observations.append({

            "epoch_id": epoch_id,

            "mjd": float(row["mjd"]),

            "image": image,

            "header": header,

            "wcs": wcs,

            "original_sources":
                detect_sources(image)

        })

    return observations


def find_valid_trajectory(observations):

    mjd = np.array([
        obs["mjd"]
        for obs in observations
    ])

    t = (mjd - mjd[0]) / 365.25

    px, py = parallax_factors(mjd)

    for attempt in range(N_ATTEMPTS):

        # Draw a distance within the
        # configured search interval.

        distance = RNG.uniform(
            MIN_DISTANCE,
            MAX_DISTANCE
        )

        parallax = (
            ARCSEC_PER_RADIAN / distance
        )

        circular_speed = (
            1296000 / distance ** 1.5
        )

        # Approximate projected orbital
        # motion with a random direction.

        angle = RNG.uniform(
            0, 2 * np.pi
        )

        speed = circular_speed * RNG.uniform(
            0.2, 1.2
        )

        vx = speed * np.cos(angle)
        vy = speed * np.sin(angle)

        # Position at the first epoch,
        # relative to the field centre.

        east0 = RNG.uniform(-150, 150)

        north0 = RNG.uniform(-150, 150)

        # Relative apparent trajectory.

        east = (
            east0
            + vx * t
            + parallax * (px - px[0])
        )

        north = (
            north0
            + vy * t
            + parallax * (py - py[0])
        )

        sky_positions = (
            REFERENCE.spherical_offsets_by(
                east * u.arcsec,
                north * u.arcsec
            )
        )

        positions = []

        valid_trajectory = True

        for k, obs in enumerate(observations):

            x, y = obs["wcs"].world_to_pixel(
                sky_positions[k]
            )

            x = float(x)
            y = float(y)

            height, width = obs["image"].shape

            if not (
                BORDER <= x < width - BORDER
                and
                BORDER <= y < height - BORDER
            ):

                valid_trajectory = False
                break

            # Check local image coverage.

            ix = int(round(x))
            iy = int(round(y))

            patch = obs["image"][
                iy-3:iy+4,
                ix-3:ix+4
            ]

            if not np.all(
                np.isfinite(patch)
            ):

                valid_trajectory = False
                break

            # Avoid injecting directly
            # on top of an existing source.

            for source in obs["original_sources"]:

                separation = np.hypot(
                    x - source["x"],
                    y - source["y"]
                )

                if separation < MIN_SOURCE_DISTANCE:

                    valid_trajectory = False
                    break

            if not valid_trajectory:
                break

            positions.append((x, y))

        if valid_trajectory:

            return {

                "distance_au": distance,

                "parallax_arcsec": parallax,

                "vx_arcsec_year": vx,

                "vy_arcsec_year": vy,

                "positions": positions,

                "east": east,

                "north": north,

                "attempts": attempt + 1

            }

    raise RuntimeError(
        "No trajectory fits inside all "
        "four images. Increase N_ATTEMPTS "
        "or use a larger field."
    )


def inject_gaussian(image, x, y):

    valid = np.isfinite(image)

    background = np.median(
        image[valid]
    )

    noise = 1.4826 * np.median(
        np.abs(
            image[valid] - background
        )
    )

    amplitude = PEAK_SNR * noise

    yy, xx = np.indices(
        image.shape
    )

    gaussian = amplitude * np.exp(

        -(
            (xx - x) ** 2
            +
            (yy - y) ** 2
        )

        / (2 * SIGMA ** 2)
    )

    injected = image.copy()

    injected[valid] += gaussian[valid]

    return injected


def main():

    observations = load_epochs()

    trajectory = find_valid_trajectory(
        observations
    )

    print("SYNTHETIC PLANET")

    print(
        "Distance:",
        round(
            trajectory["distance_au"], 2
        ),
        "AU"
    )

    print(
        "Parallax:",
        round(
            trajectory["parallax_arcsec"], 3
        ),
        "arcsec"
    )

    print(
        "Velocity:",
        round(
            trajectory["vx_arcsec_year"], 3
        ),
        round(
            trajectory["vy_arcsec_year"], 3
        ),
        "arcsec/year"
    )

    print(
        "Attempts required:",
        trajectory["attempts"]
    )

    records = []
    truth = []

    fig, axes = plt.subplots(
        2, 2,
        figsize=(12, 10)
    )

    axes = axes.ravel()

    for k, obs in enumerate(observations):

        epoch_id = obs["epoch_id"]

        x_true, y_true = (
            trajectory["positions"][k]
        )

        injected = inject_gaussian(

            obs["image"],

            x_true,

            y_true
        )

        fits.writeto(

            OUTPUT_DIR
            / f"epoch_{epoch_id}_injected.fits",

            injected,

            obs["header"],

            overwrite=True
        )

        sources = detect_sources(
            injected
        )

        if not sources:

            raise RuntimeError(
                f"No sources in epoch {epoch_id}"
            )

        distances = np.array([

            np.hypot(
                source["x"] - x_true,
                source["y"] - y_true
            )

            for source in sources

        ])

        closest_index = int(
            np.argmin(distances)
        )

        recovery_error = float(
            distances[closest_index]
        )

        # Recovery must be within one
        # pixel of the injected position.

        recovered = recovery_error <= 1.0

        true_source_id = (

            f"E{epoch_id}_{closest_index+1:03d}"

            if recovered else None
        )

        truth.append({

            "epoch_id": epoch_id,

            "mjd": obs["mjd"],

            "true_x": x_true,

            "true_y": y_true,

            "true_distance_au":
                trajectory["distance_au"],

            "true_vx":
                trajectory["vx_arcsec_year"],

            "true_vy":
                trajectory["vy_arcsec_year"],

            "recovered": recovered,

            "recovered_source_id":
                true_source_id,

            "centroid_error_pixels":
                recovery_error

        })

        print(
            f"\nEpoch {epoch_id}"
        )

        print(
            f"True position: "
            f"({x_true:.3f}, {y_true:.3f})"
        )

        print(
            "Recovered source:",
            true_source_id
        )

        print(
            f"Centroid error: "
            f"{recovery_error:.3f} pixels"
        )

        for j, source in enumerate(
            sources, 1
        ):

            source_id = (
                f"E{epoch_id}_{j:03d}"
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

                "epoch_id": epoch_id,

                "source_id": source_id,

                "mjd": obs["mjd"],

                "x": source["x"],

                "y": source["y"],

                "ra": float(ra),

                "dec": float(dec),

                "east_arcsec": east.arcsec,

                "north_arcsec": north.arcsec,

                "peak_snr":
                    source["peak_snr"],

                "area": source["area"],

                "sigma_x": source["sigma_x"],

                "sigma_y": source["sigma_y"]

            })

        ax = axes[k]

        valid = np.isfinite(injected)

        ax.imshow(

            injected,

            origin="lower",

            cmap="gray",

            vmin=np.percentile(
                injected[valid], 5
            ),

            vmax=np.percentile(
                injected[valid], 99
            )

        )

        ax.scatter(

            x_true,

            y_true,

            s=150,

            facecolors="none",

            edgecolors="yellow",

            linewidths=2,

            label="Injected position"

        )

        if recovered:

            source = sources[closest_index]

            ax.plot(

                source["x"],

                source["y"],

                "+",

                color="cyan",

                markersize=12,

                label="Recovered centroid"

            )

        ax.set_title(
            f"Epoch {epoch_id}"
        )

        ax.legend(
            loc="upper right",
            fontsize=8
        )

    catalogue = pd.DataFrame(records)

    truth_df = pd.DataFrame(truth)

    catalogue.to_csv(

        OUTPUT_DIR / "source_catalogue.csv",

        index=False
    )

    truth_df.to_csv(

        OUTPUT_DIR / "ground_truth.csv",

        index=False
    )

    plt.suptitle(
        "PlanetHunter: Synthetic Planet Recovery"
    )

    plt.tight_layout()

    plt.savefig(

        OUTPUT_DIR / "recovery_preview.png",

        dpi=150
    )

    # --------------------------------
    # Run our existing trajectory engine
    # --------------------------------

    epoch_ids = tuple(
        sorted(
            catalogue["epoch_id"].unique()
        )
    )

    candidates, n_tracks = (
        evaluate_epoch_set(
            catalogue,
            epoch_ids
        )
    )

    candidates = candidates.reset_index(
        drop=True
    )

    candidates.insert(

        0,

        "rank",

        np.arange(
            1, len(candidates) + 1
        )
    )

    # Identify the correct track using
    # our independently known ground truth.

    correct = np.ones(
        len(candidates),
        dtype=bool
    )

    for _, row in truth_df.iterrows():

        epoch = int(row["epoch_id"])

        if not row["recovered"]:

            correct[:] = False
            break

        correct &= (

            candidates[f"source_E{epoch}"]

            == row["recovered_source_id"]

        ).to_numpy()

    candidates["is_true_track"] = correct

    candidates.to_csv(

        OUTPUT_DIR / "moving_candidates.csv",

        index=False
    )

    print("\nRECOVERY SUMMARY")

    print(
        "Recovered detections:",
        int(truth_df["recovered"].sum()),
        "/",
        len(observations)
    )

    print(
        "Trajectories evaluated:",
        n_tracks
    )

    true_track = candidates[
        candidates["is_true_track"]
    ]

    if not true_track.empty:

        best = true_track.iloc[0]

        fitted_distance = float(
            best["distance_au"]
        )

        true_distance = (
            trajectory["distance_au"]
        )

        distance_error = (

            abs(
                fitted_distance - true_distance
            )

            / true_distance

            * 100

        )

        print(
            "Correct track rank:",
            int(best["rank"])
        )

        print(
            "True distance:",
            round(true_distance, 2),
            "AU"
        )

        print(
            "Recovered distance:",
            round(fitted_distance, 2),
            "AU"
        )

        print(
            "Distance error:",
            round(distance_error, 3),
            "%"
        )

        print(
            "Parallax RMS:",
            round(best["parallax_rms"], 3),
            "arcsec"
        )

        print(
            "Passes screen:",
            best["passes_screen"]
        )

    else:

        print(
            "Correct track was not found "
            "among retained candidates."
        )

    print(
        "\nResults saved in:",
        OUTPUT_DIR
    )

    plt.show()


if __name__ == "__main__":
    main()
