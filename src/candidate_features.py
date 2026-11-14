
import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist

from astropy.coordinates import SkyCoord
import astropy.units as u

from motion_models import (
    design_matrices,
    fit_batch,
    parallax_factors,
    REFERENCE_RA,
    REFERENCE_DEC
)


REFERENCE = SkyCoord(
    ra=REFERENCE_RA * u.deg,
    dec=REFERENCE_DEC * u.deg
)

STATIONARY_RADIUS = 6.1

BORDER = 5


NEW_FEATURES = [

    "stationary_member_fraction",

    "mean_stationary_support",

    "max_stationary_support",

    "mean_nearest_counterpart",

    "max_fit_residual",

    "dof_adjusted_residual",

    "time_baseline_days",

    "escape_speed_ratio",

    "missing_inside_field",

    "missing_outside_field",

    "missing_invalid_region"
]


# --------------------------------------------------
# PREPARE SOURCE CATALOGUE
# --------------------------------------------------

def prepare_catalogue(catalogue):

    catalogue = catalogue.copy()

    if not {

        "east_arcsec",
        "north_arcsec"

    }.issubset(catalogue.columns):

        coords = SkyCoord(

            ra=catalogue["ra"].to_numpy()*u.deg,

            dec=catalogue["dec"].to_numpy()*u.deg
        )

        east, north = (
            REFERENCE.spherical_offsets_to(
                coords
            )
        )

        catalogue["east_arcsec"] = (
            east.arcsec
        )

        catalogue["north_arcsec"] = (
            north.arcsec
        )

    return catalogue


# --------------------------------------------------
# STATIONARY SOURCE EVIDENCE
# --------------------------------------------------

def stationary_evidence(catalogue):

    catalogue = catalogue.copy()

    n = len(catalogue)

    support = np.zeros(n, dtype=int)

    nearest = np.full(n, np.inf)

    positions = catalogue[[

        "east_arcsec",

        "north_arcsec"

    ]].to_numpy()

    epoch_ids = catalogue[
        "epoch_id"
    ].to_numpy()

    unique_epochs = np.unique(
        epoch_ids
    )

    for epoch1 in unique_epochs:

        idx1 = np.flatnonzero(
            epoch_ids == epoch1
        )

        for epoch2 in unique_epochs:

            if epoch1 == epoch2:

                continue

            idx2 = np.flatnonzero(
                epoch_ids == epoch2
            )

            if len(idx2) == 0:

                continue

            distances = cdist(

                positions[idx1],

                positions[idx2]
            )

            closest = distances.min(
                axis=1
            )

            nearest[idx1] = np.minimum(

                nearest[idx1],

                closest
            )

            # Require mutual nearest
            # neighbours for support.

            best2 = distances.argmin(
                axis=1
            )

            best1 = distances.argmin(
                axis=0
            )

            mutual = (

                best1[best2]

                == np.arange(len(idx1))
            )

            matched = (

                (closest <= STATIONARY_RADIUS)

                & mutual
            )

            support[idx1] += matched.astype(
                int
            )

    catalogue["stationary_support"] = (
        support
    )

    # Finite fallback when no other
    # epoch contains any sources.

    catalogue["nearest_counterpart"] = (

        np.where(

            np.isfinite(nearest),

            nearest,

            360.0
        )
    )

    return catalogue


# --------------------------------------------------
# PREDICT CANDIDATE POSITIONS AT ALL EPOCHS
# --------------------------------------------------

def predict_all_epochs(
    observed,
    selected_mjd,
    all_mjd,
    fit
):

    # Model:
    #
    # position(t) =
    # intercept + velocity*t + parallax*P(t)

    selected_t = (

        selected_mjd - selected_mjd[0]

    ) / 365.25

    all_t = (

        all_mjd - selected_mjd[0]

    ) / 365.25

    selected_px, selected_py = (
        parallax_factors(selected_mjd)
    )

    all_px, all_py = (
        parallax_factors(all_mjd)
    )

    vx = np.asarray(

        fit["vx_arcsec_year"]

    ).reshape(-1, 1)

    vy = np.asarray(

        fit["vy_arcsec_year"]

    ).reshape(-1, 1)

    parallax = np.asarray(

        fit["parallax_arcsec"]

    ).reshape(-1, 1)

    east0 = np.mean(

        observed[:, :, 0]

        - vx*selected_t[None, :]

        - parallax*selected_px[None, :],

        axis=1,

        keepdims=True
    )

    north0 = np.mean(

        observed[:, :, 1]

        - vy*selected_t[None, :]

        - parallax*selected_py[None, :],

        axis=1,

        keepdims=True
    )

    east = (

        east0

        + vx*all_t[None, :]

        + parallax*all_px[None, :]
    )

    north = (

        north0

        + vy*all_t[None, :]

        + parallax*all_py[None, :]
    )

    return east, north


# --------------------------------------------------
# VISIBILITY FEATURES
# --------------------------------------------------

def missing_epoch_features(
    east,
    north,
    selected_epochs,
    observations
):

    n = east.shape[0]

    missing_inside = np.zeros(
        n, dtype=int
    )

    missing_outside = np.zeros(
        n, dtype=int
    )

    missing_invalid = np.zeros(
        n, dtype=int
    )

    for j, obs in enumerate(
        observations
    ):

        epoch_id = obs["epoch_id"]

        if epoch_id in selected_epochs:

            continue

        coords = (
            REFERENCE.spherical_offsets_by(

                east[:, j]*u.arcsec,

                north[:, j]*u.arcsec
            )
        )

        x, y = obs["wcs"].world_to_pixel(
            coords
        )

        image = obs["image"]

        height, width = image.shape

        finite = (

            np.isfinite(x)

            & np.isfinite(y)
        )

        inside = (

            finite

            & (x >= 0)

            & (x < width)

            & (y >= 0)

            & (y < height)
        )

        usable = (

            inside

            & (x >= BORDER)

            & (x < width-BORDER)

            & (y >= BORDER)

            & (y < height-BORDER)
        )

        for i in np.flatnonzero(usable):

            xi = int(round(x[i]))

            yi = int(round(y[i]))

            patch = image[

                yi-3:yi+4,

                xi-3:xi+4
            ]

            usable[i] = (

                patch.shape == (7, 7)

                and np.isfinite(
                    patch
                ).all()
            )

        missing_inside += usable.astype(
            int
        )

        missing_outside += (

            finite & ~inside

        ).astype(int)

        missing_invalid += (

            (~finite)

            | (inside & ~usable)

        ).astype(int)

    return (

        missing_inside,

        missing_outside,

        missing_invalid
    )


# --------------------------------------------------
# EXTRACT FEATURES FOR RETAINED CANDIDATES
# --------------------------------------------------

def augment_candidates(
    candidates,
    catalogue,
    observations
):

    candidates = candidates.copy()

    catalogue = prepare_catalogue(
        catalogue
    )

    catalogue = stationary_evidence(
        catalogue
    )

    lookup = catalogue.set_index(
        "source_id"
    )

    observations = sorted(

        observations,

        key=lambda obs: obs["mjd"]
    )

    all_mjd = np.array([

        obs["mjd"]

        for obs in observations
    ])

    for feature in NEW_FEATURES:

        candidates[feature] = np.nan

    # Process candidates in batches
    # sharing the same observation epochs.

    for epoch_string, group in candidates.groupby(

        "epoch_ids",

        sort=False
    ):

        selected_epochs = [

            int(e)

            for e in str(
                epoch_string
            ).split(";")
        ]

        selected_epochs.sort(

            key=lambda e: next(

                obs["mjd"]

                for obs in observations

                if obs["epoch_id"] == e
            )
        )

        source_columns = [

            f"source_E{e}"

            for e in selected_epochs
        ]

        source_ids = group[
            source_columns
        ].to_numpy()

        n_tracks = len(group)

        n_epochs = len(
            selected_epochs
        )

        members = lookup.loc[

            source_ids.reshape(-1)
        ]

        observed = members[[

            "east_arcsec",

            "north_arcsec"

        ]].to_numpy().reshape(

            n_tracks,

            n_epochs,

            2
        )

        support = members[

            "stationary_support"

        ].to_numpy().reshape(

            n_tracks,

            n_epochs
        )

        nearest = members[

            "nearest_counterpart"

        ].to_numpy().reshape(

            n_tracks,

            n_epochs
        )

        # At least two other epochs
        # support a persistent source.

        persistent_fraction = (

            support >= 2

        ).mean(axis=1)

        selected_mjd = np.array([

            next(

                obs["mjd"]

                for obs in observations

                if obs["epoch_id"] == e
            )

            for e in selected_epochs
        ])

        matrices = design_matrices(
            selected_mjd
        )

        fitted = fit_batch(

            observed.reshape(
                n_tracks, -1
            ),

            matrices
        )

        prediction = np.asarray(

            fitted["prediction"]

        ).reshape(

            n_tracks,

            n_epochs,

            2
        )

        residual = observed - prediction

        radial_residual = np.linalg.norm(

            residual,

            axis=2
        )

        max_residual = radial_residual.max(
            axis=1
        )

        # Five fitted parameters.
        # This is not a calibrated chi-square.

        dof = max(
            2*n_epochs - 5,
            1
        )

        adjusted_residual = np.sqrt(

            np.sum(

                residual**2,

                axis=(1, 2)

            ) / dof
        )

        east, north = predict_all_epochs(

            observed,

            selected_mjd,

            all_mjd,

            fitted
        )

        inside, outside, invalid = (

            missing_epoch_features(

                east,

                north,

                selected_epochs,

                observations
            )
        )

        values = {

            "stationary_member_fraction":
                persistent_fraction,

            "mean_stationary_support":
                support.mean(axis=1),

            "max_stationary_support":
                support.max(axis=1),

            "mean_nearest_counterpart":
                nearest.mean(axis=1),

            "max_fit_residual":
                max_residual,

            "dof_adjusted_residual":
                adjusted_residual,

            "time_baseline_days":
                selected_mjd[-1]
                - selected_mjd[0],

            "escape_speed_ratio":

                group[
                    "circular_speed_ratio"
                ].to_numpy()

                / np.sqrt(2),

            "missing_inside_field":
                inside,

            "missing_outside_field":
                outside,

            "missing_invalid_region":
                invalid
        }

        for feature, value in values.items():

            candidates.loc[

                group.index,

                feature

            ] = value

    return candidates
