
import numpy as np
import pandas as pd

from pathlib import Path

from astropy.coordinates import SkyCoord
import astropy.units as u

from motion_models import (
    parallax_factors,
    REFERENCE_RA,
    REFERENCE_DEC,
    ARCSEC_PER_RADIAN
)

from synthetic_recovery import load_epochs


RNG = np.random.default_rng(2026)

N_PER_BIN = 2000

DISTANCE_BINS = [
    (200, 400),
    (400, 600),
    (600, 800),
    (800, 1000),
    (1000, 1200),
    (1200, 1500)
]

BORDER = 5

REFERENCE = SkyCoord(
    ra=REFERENCE_RA * u.deg,
    dec=REFERENCE_DEC * u.deg
)


def main():

    observations = load_epochs()

    mjd = np.array([
        obs["mjd"]
        for obs in observations
    ])

    t = (mjd - mjd[0]) / 365.25

    px, py = parallax_factors(mjd)

    results = []

    for low, high in DISTANCE_BINS:

        n = N_PER_BIN

        # Sample distances uniformly
        # within the current interval.

        distance = RNG.uniform(
            low, high, n
        )

        parallax = (
            ARCSEC_PER_RADIAN / distance
        )

        # Approximate circular angular
        # speed in arcsec/year.

        circular_speed = (
            1296000 / distance**1.5
        )

        speed = circular_speed * RNG.uniform(
            0.2, 1.2, n
        )

        angle = RNG.uniform(
            0, 2*np.pi, n
        )

        vx = speed * np.cos(angle)
        vy = speed * np.sin(angle)

        # Anchor the initial apparent
        # position near the first field.

        east0 = RNG.uniform(
            -150, 150, n
        )

        north0 = RNG.uniform(
            -150, 150, n
        )

        # Predict all positions using
        # broadcasting: N objects x epochs.

        east = (
            east0[:, None]
            + vx[:, None] * t[None, :]
            + parallax[:, None]
            * (px - px[0])[None, :]
        )

        north = (
            north0[:, None]
            + vy[:, None] * t[None, :]
            + parallax[:, None]
            * (py - py[0])[None, :]
        )

        visible = np.zeros(
            (n, len(observations)),
            dtype=bool
        )

        for j, obs in enumerate(
            observations
        ):

            coords = (
                REFERENCE.spherical_offsets_by(
                    east[:, j] * u.arcsec,
                    north[:, j] * u.arcsec
                )
            )

            x, y = obs["wcs"].world_to_pixel(
                coords
            )

            image = obs["image"]

            height, width = image.shape

            inside = (
                np.isfinite(x)
                & np.isfinite(y)
                & (x >= BORDER)
                & (x < width - BORDER)
                & (y >= BORDER)
                & (y < height - BORDER)
            )

            # Also require a finite image
            # patch around the source.

            for i in np.flatnonzero(inside):

                xi = int(round(x[i]))
                yi = int(round(y[i]))

                patch = image[
                    yi-3:yi+4,
                    xi-3:xi+4
                ]

                visible[i, j] = (
                    patch.shape == (7, 7)
                    and np.isfinite(patch).all()
                )

        n_visible = visible.sum(axis=1)

        three = np.sum(n_visible == 3)

        four = np.sum(n_visible == 4)

        usable = np.sum(n_visible >= 3)

        row = {

            "distance_min_au": low,

            "distance_max_au": high,

            "simulated_objects": n,

            "zero_epochs": int(
                np.sum(n_visible == 0)
            ),

            "one_epoch": int(
                np.sum(n_visible == 1)
            ),

            "two_epochs": int(
                np.sum(n_visible == 2)
            ),

            "three_epochs": int(three),

            "four_epochs": int(four),

            "usable_trajectories": int(
                usable
            ),

            "observability_percent": (
                100 * usable / n
            )
        }

        results.append(row)

        print(
            f"\nDistance: {low}-{high} AU"
        )

        print(
            "Exactly three epochs:",
            three
        )

        print(
            "Exactly four epochs:",
            four
        )

        print(
            "Usable trajectories:",
            usable,
            "/",
            n
        )

        print(
            "Observability:",
            f"{100*usable/n:.2f}%"
        )

    df = pd.DataFrame(results)

    output = Path(
        "data/ml/observability.csv"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output,
        index=False
    )

    print("\nFINAL OBSERVABILITY SUMMARY")

    print(
        df.to_string(index=False)
    )

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
