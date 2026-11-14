
# src/live_discovery.py

from itertools import combinations, product

import numpy as np
import pandas as pd

from astropy import units as u
from astropy.time import Time
from astropy.coordinates import (
    SkyCoord,
    get_body_barycentric,
)

from source_detection import detect_sources
from sky_explorer import make_reference_grid


ARCSEC_PER_RADIAN = 206264.806

MIN_DISTANCE = 200
MAX_DISTANCE = 1500

MAX_SOURCES = 20
MAX_EPOCHS = 4


# ============================================================
# SELECT OBSERVATIONS
# ============================================================

def select_epochs(sequence):

    mjd = sequence["mjd"].to_numpy(dtype=float)

    ordered = np.argsort(mjd)

    if len(ordered) <= MAX_EPOCHS:
        return ordered.tolist()

    selected = [
        int(ordered[0]),
        int(ordered[-1])
    ]

    remaining = set(ordered) - set(selected)

    while len(selected) < MAX_EPOCHS:

        best = max(
            remaining,
            key=lambda i: min(
                abs(mjd[i] - mjd[j])
                for j in selected
            )
        )

        selected.append(int(best))
        remaining.remove(best)

    return sorted(
        selected,
        key=lambda i: mjd[i]
    )


# ============================================================
# BUILD SOURCE CATALOGUE
# ============================================================

def build_catalogue(data, selected):

    images = data["images"]
    sequence = data["sequence"]

    wcs, _ = make_reference_grid(
        data["ra"],
        data["dec"],
        data["fov"]
    )

    reference = SkyCoord(
        data["ra"] * u.deg,
        data["dec"] * u.deg,
        frame="icrs"
    )

    catalogue = {}

    for epoch in selected:

        detections = detect_sources(
            images[epoch]
        )

        detections = sorted(
            detections,
            key=lambda s: float(
                s.get("peak_snr", 0)
            ),
            reverse=True
        )[:MAX_SOURCES]

        rows = []

        for index, source in enumerate(
            detections
        ):

            x = float(source["x"])
            y = float(source["y"])

            position = wcs.pixel_to_world(
                x,
                y
            )

            east, north = (
                reference.spherical_offsets_to(
                    position
                )
            )

            rows.append({

                "source_id":
                    f"E{epoch+1}_{index+1:03d}",

                "epoch": epoch,

                "date":
                    str(
                        sequence.iloc[epoch]["date"]
                    )[:10],

                "x": x,

                "y": y,

                "ra": position.ra.deg,

                "dec": position.dec.deg,

                "east":
                    east.to_value(u.arcsec),

                "north":
                    north.to_value(u.arcsec),

                "peak_snr":
                    float(
                        source.get("peak_snr", 0)
                    ),

                "stationary_support": 0
            })

        catalogue[epoch] = pd.DataFrame(rows)

    return catalogue


# ============================================================
# STATIONARY SOURCE ASSOCIATION
# ============================================================

def associate_stationary(catalogue):

    for first, second in combinations(
        catalogue,
        2
    ):

        a = catalogue[first]
        b = catalogue[second]

        if a.empty or b.empty:
            continue

        pa = a[
            ["east", "north"]
        ].to_numpy()

        pb = b[
            ["east", "north"]
        ].to_numpy()

        distance = np.linalg.norm(
            pa[:, None, :] - pb[None, :, :],
            axis=2
        )

        nearest_b = distance.argmin(axis=1)
        nearest_a = distance.argmin(axis=0)

        for i, j in enumerate(nearest_b):

            if (
                nearest_a[j] == i
                and distance[i, j] <= 6.1
            ):

                a.loc[
                    i,
                    "stationary_support"
                ] += 1

                b.loc[
                    j,
                    "stationary_support"
                ] += 1

    return catalogue


# ============================================================
# PARALLAX FACTORS
# ============================================================

def parallax_factors(mjd, ra, dec):

    time = Time(
        mjd,
        format="mjd"
    )

    earth = get_body_barycentric(
        "earth",
        time
    ).xyz.to_value(u.au).T

    sun = get_body_barycentric(
        "sun",
        time
    ).xyz.to_value(u.au).T

    observer = earth - sun

    a = np.radians(ra)
    d = np.radians(dec)

    east_vector = np.array([
        -np.sin(a),
        np.cos(a),
        0
    ])

    north_vector = np.array([
        -np.cos(a) * np.sin(d),
        -np.sin(a) * np.sin(d),
        np.cos(d)
    ])

    px = -observer @ east_vector
    py = -observer @ north_vector

    return px, py


# ============================================================
# CONSTRUCT MOTION MODELS
# ============================================================

def design_matrices(mjd, ra, dec):

    mjd = np.asarray(
        mjd,
        dtype=float
    )

    t = (
        mjd - mjd[0]
    ) / 365.25

    n = len(t)

    stationary = np.zeros(
        (2*n, 2)
    )

    stationary[0::2, 0] = 1
    stationary[1::2, 1] = 1

    linear = np.zeros(
        (2*n, 4)
    )

    linear[:, :2] = stationary

    linear[0::2, 2] = t
    linear[1::2, 3] = t

    px, py = parallax_factors(
        mjd,
        ra,
        dec
    )

    parallax = np.zeros(
        2*n
    )

    parallax[0::2] = px
    parallax[1::2] = py

    return stationary, linear, parallax


# ============================================================
# BATCH TRAJECTORY FITTING
# ============================================================

def fit_batch(
    observations,
    stationary,
    linear,
    parallax
):

    n = observations.shape[1] // 2

    stationary_pinv = np.linalg.pinv(
        stationary
    )

    linear_pinv = np.linalg.pinv(
        linear
    )

    stationary_parameters = (
        observations
        @ stationary_pinv.T
    )

    stationary_prediction = (
        stationary_parameters
        @ stationary.T
    )

    linear_parameters = (
        observations
        @ linear_pinv.T
    )

    linear_prediction = (
        linear_parameters
        @ linear.T
    )

    stationary_residual = (
        observations
        - stationary_prediction
    )

    linear_residual = (
        observations
        - linear_prediction
    )

    # Remove the component of parallax
    # already explained by linear motion.

    q = parallax - linear @ (
        linear_pinv @ parallax
    )

    leverage = float(q @ q)

    if leverage < 1e-10:
        return None

    raw_parallax = (
        linear_residual @ q
    ) / leverage

    minimum_parallax = (
        ARCSEC_PER_RADIAN
        / MAX_DISTANCE
    )

    maximum_parallax = (
        ARCSEC_PER_RADIAN
        / MIN_DISTANCE
    )

    fitted_parallax = np.clip(
        raw_parallax,
        minimum_parallax,
        maximum_parallax
    )

    at_boundary = (
        (raw_parallax <= minimum_parallax)
        |
        (raw_parallax >= maximum_parallax)
    )

    corrected = (
        observations
        -
        fitted_parallax[:, None]
        * parallax[None, :]
    )

    parameters = (
        corrected @ linear_pinv.T
    )

    prediction = (

        parameters @ linear.T

        +

        fitted_parallax[:, None]
        * parallax[None, :]
    )

    residual = (
        observations - prediction
    )

    def rms(array):

        return np.sqrt(
            np.sum(
                array**2,
                axis=1
            ) / n
        )

    distance = (
        ARCSEC_PER_RADIAN
        / fitted_parallax
    )

    speed = np.hypot(
        parameters[:, 2],
        parameters[:, 3]
    )

    circular_speed = (
        1296000
        / distance**1.5
    )

    speed_ratio = (
        speed / circular_speed
    )

    return {

        "stationary_rms":
            rms(stationary_residual),

        "linear_rms":
            rms(linear_residual),

        "parallax_rms":
            rms(residual),

        "distance_au":
            distance,

        "speed_arcsec_year":
            speed,

        "circular_speed_ratio":
            speed_ratio,

        "escape_speed_ratio":
            speed_ratio / np.sqrt(2),

        "parallax_at_boundary":
            at_boundary,

        "parallax_leverage":
            leverage,

        "prediction":
            prediction,

        "parameters":
            parameters,

        "parallax_arcsec":
            fitted_parallax
    }


# ============================================================
# EVALUATE ONE EPOCH SUBSET
# ============================================================

def evaluate_subset(
    catalogue,
    epochs,
    data
):

    tables = [
        catalogue[e]
        for e in epochs
    ]

    if any(t.empty for t in tables):
        return []

    n = len(epochs)

    indices = np.array(

        list(
            product(
                *[
                    range(len(t))
                    for t in tables
                ]
            )
        ),

        dtype=int
    )

    observations = np.empty(
        (len(indices), 2*n)
    )

    for k, table in enumerate(tables):

        positions = table[
            ["east", "north"]
        ].to_numpy()

        observations[
            :,
            2*k:2*k+2
        ] = positions[indices[:, k]]

    mjd = [

        float(
            data["sequence"].iloc[e]["mjd"]
        )

        for e in epochs
    ]

    stationary, linear, parallax = (
        design_matrices(
            mjd,
            data["ra"],
            data["dec"]
        )
    )

    fits = fit_batch(
        observations,
        stationary,
        linear,
        parallax
    )

    if fits is None:
        return []

    pr = fits["parallax_rms"]
    sr = fits["stationary_rms"]
    lr = fits["linear_rms"]

    speed_ratio = (
        fits["circular_speed_ratio"]
    )

    boundary = (
        fits["parallax_at_boundary"]
    )

    # Original exploratory physics screen.

    passes = (

        (pr <= 3.0)

        & (sr >= 3.0)

        & (speed_ratio <= 2.0)

        & (~boundary)
    )

    # Penalize associations made entirely
    # from persistent background sources.

    support = np.zeros(
        (len(indices), n)
    )

    for k, table in enumerate(tables):

        support[:, k] = (

            table[
                "stationary_support"
            ].to_numpy()

            [indices[:, k]]
        )

    stationary_fraction = np.mean(
        support >= 2,
        axis=1
    )

    score = (

        pr

        + 2 * np.maximum(
            speed_ratio - 1.5,
            0
        )

        + 3 * boundary

        + np.maximum(
            3 - sr,
            0
        )

        + 3 * stationary_fraction
    )

    # Prefer passing candidates, then
    # lower physical ranking scores.

    order = np.lexsort(
        (score, ~passes)
    )[:100]

    rows = []

    for i in order:

        members = [

            tables[k].iloc[
                indices[i, k]
            ].to_dict()

            for k in range(n)
        ]

        rows.append({

            "n_epochs": n,

            "epoch_indices":
                list(epochs),

            "distance_au":
                float(
                    fits["distance_au"][i]
                ),

            "parallax_rms":
                float(pr[i]),

            "linear_rms":
                float(lr[i]),

            "stationary_rms":
                float(sr[i]),

            "circular_speed_ratio":
                float(speed_ratio[i]),

            "escape_speed_ratio":
                float(
                    fits[
                        "escape_speed_ratio"
                    ][i]
                ),

            "stationary_member_fraction":
                float(
                    stationary_fraction[i]
                ),

            "parallax_at_boundary":
                bool(boundary[i]),

            "parallax_leverage":
                fits["parallax_leverage"],

            "passes_screen":
                bool(passes[i]),

            "physics_score":
                float(score[i]),

            "members":
                members,

            "parameters":
                fits[
                    "parameters"
                ][i].tolist(),

            "parallax_arcsec":
                float(
                    fits[
                        "parallax_arcsec"
                    ][i]
                ),

            "reference_mjd":
                mjd[0]
        })

    return rows


# ============================================================
# MAIN LIVE DISCOVERY PIPELINE
# ============================================================

def run_discovery(data):

    if len(data["sequence"]) < 3:

        raise ValueError(
            "Discovery requires at least "
            "three observations."
        )

    selected = select_epochs(
        data["sequence"]
    )

    catalogue = build_catalogue(
        data,
        selected
    )

    catalogue = associate_stationary(
        catalogue
    )

    usable = [

        e for e in selected

        if not catalogue[e].empty
    ]

    if len(usable) < 3:

        raise ValueError(
            "Fewer than three observations "
            "contain usable detected sources."
        )

    candidates = []

    if len(usable) == 4:

        candidates.extend(

            evaluate_subset(
                catalogue,
                usable,
                data
            )
        )

    for subset in combinations(
        usable,
        3
    ):

        candidates.extend(

            evaluate_subset(
                catalogue,
                subset,
                data
            )
        )

    candidates.sort(

        key=lambda c: (

            not c["passes_screen"],

            -c["n_epochs"],

            c["physics_score"]
        )
    )

    candidates = candidates[:100]

    for i, candidate in enumerate(
        candidates
    ):

        candidate["candidate_id"] = (
            f"L{i+1:04d}"
        )

    sources = pd.concat(

        [
            table
            for table in catalogue.values()
            if not table.empty
        ],

        ignore_index=True
    )

    return {

        "candidates": candidates,

        "sources": sources,

        "selected_epochs": selected,

        "n_sources": len(sources)
    }


# ============================================================
# PREDICT CANDIDATE POSITIONS AT EVERY EPOCH
# ============================================================

def predict_candidate(
    candidate,
    data
):

    sequence = data["sequence"]

    mjd = sequence[
        "mjd"
    ].to_numpy(dtype=float)

    t = (

        mjd
        - candidate["reference_mjd"]

    ) / 365.25

    x0, y0, vx, vy = (
        candidate["parameters"]
    )

    px, py = parallax_factors(

        mjd,

        data["ra"],

        data["dec"]
    )

    parallax = (
        candidate["parallax_arcsec"]
    )

    east = (
        x0 + vx*t + parallax*px
    )

    north = (
        y0 + vy*t + parallax*py
    )

    reference = SkyCoord(

        data["ra"] * u.deg,

        data["dec"] * u.deg,

        frame="icrs"
    )

    positions = (
        reference.spherical_offsets_by(

            east * u.arcsec,

            north * u.arcsec
        )
    )

    wcs, _ = make_reference_grid(

        data["ra"],

        data["dec"],

        data["fov"]
    )

    x, y = wcs.world_to_pixel(
        positions
    )

    return list(
        zip(x, y)
    )
