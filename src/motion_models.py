
import numpy as np

from astropy.time import Time
from astropy.coordinates import (
    get_body_barycentric,
    solar_system_ephemeris
)
import astropy.units as u


ARCSEC_PER_RADIAN = 206264.806

REFERENCE_RA = 337.8783221
REFERENCE_DEC = -5.1891588


def parallax_factors(
    mjd,
    ra=REFERENCE_RA,
    dec=REFERENCE_DEC
):
    """
    Calculate heliocentric Earth-based
    parallax factors in a local tangent plane.

    Earth centre approximates the position
    of SPHEREx for this prototype.
    """

    times = Time(
        np.asarray(mjd),
        format="mjd",
        scale="utc"
    )

    with solar_system_ephemeris.set("builtin"):

        earth = get_body_barycentric(
            "earth", times
        )

        sun = get_body_barycentric(
            "sun", times
        )

    observer = (
        earth - sun
    ).xyz.to_value(u.au).T

    alpha = np.radians(ra)
    delta = np.radians(dec)

    east = np.array([
        -np.sin(alpha),
        np.cos(alpha),
        0.0
    ])

    north = np.array([
        -np.cos(alpha) * np.sin(delta),
        -np.sin(alpha) * np.sin(delta),
        np.cos(delta)
    ])

    px = -(observer @ east)
    py = -(observer @ north)

    return px, py


def design_matrices(
    mjd,
    ra=REFERENCE_RA,
    dec=REFERENCE_DEC
):
    """
    Construct the matrices for stationary,
    linear and parallax-aware motion models.
    """

    mjd = np.asarray(mjd, dtype=float)

    # Time measured in years relative
    # to the first observation.

    t = (mjd - mjd[0]) / 365.25

    px, py = parallax_factors(
        mjd, ra, dec
    )

    n = len(t)

    stationary = np.zeros((2 * n, 2))

    stationary[0::2, 0] = 1
    stationary[1::2, 1] = 1

    linear = np.zeros((2 * n, 4))

    linear[:, :2] = stationary

    linear[0::2, 2] = t
    linear[1::2, 3] = t

    parallax = np.zeros((2 * n, 5))

    parallax[:, :4] = linear

    parallax[0::2, 4] = px
    parallax[1::2, 4] = py

    return stationary, linear, parallax


def fit_batch(
    observations,
    matrices,
    min_distance=200,
    max_distance=1500
):
    """
    Fit many candidate tracks simultaneously.

    observations shape:
        (number_of_tracks, 2 * number_of_epochs)

    Each row:
        [x1, y1, x2, y2, ...]

    Coordinates are tangent-plane arcseconds.

    Parallax is constrained to the selected
    heliocentric distance range.
    """

    observations = np.asarray(
        observations, dtype=float
    )

    stationary, linear, parallax = matrices

    n_epochs = observations.shape[1] // 2

    # ---------------------------
    # Stationary model
    # ---------------------------

    stationary_params = (
        observations
        @ np.linalg.pinv(stationary).T
    )

    stationary_prediction = (
        stationary_params @ stationary.T
    )

    stationary_residual = (
        observations - stationary_prediction
    )

    stationary_rms = np.sqrt(
        np.sum(
            stationary_residual ** 2,
            axis=1
        ) / n_epochs
    )

    # ---------------------------
    # Linear motion model
    # ---------------------------

    linear_inverse = np.linalg.pinv(linear)

    linear_params = (
        observations @ linear_inverse.T
    )

    linear_prediction = (
        linear_params @ linear.T
    )

    linear_residual = (
        observations - linear_prediction
    )

    linear_rms = np.sqrt(
        np.sum(
            linear_residual ** 2,
            axis=1
        ) / n_epochs
    )

    # ---------------------------
    # Parallax-aware model
    # ---------------------------

    p = parallax[:, 4]

    # Remove the part of the parallax
    # signal that can be explained by
    # position and linear motion.

    p_residual = (
        p - linear @ (linear_inverse @ p)
    )

    denominator = np.dot(
        p_residual, p_residual
    )

    if denominator < 1e-10:

        raise ValueError(
            "Observation geometry cannot "
            "constrain parallax reliably"
        )

    raw_pi = (
        linear_residual @ p_residual
    ) / denominator

    min_pi = (
        ARCSEC_PER_RADIAN / max_distance
    )

    max_pi = (
        ARCSEC_PER_RADIAN / min_distance
    )

    # Exact bounded least-squares solution
    # for the single parallax parameter.

    fitted_pi = np.clip(
        raw_pi,
        min_pi,
        max_pi
    )

    # Refit position and velocity after
    # fixing the bounded parallax.

    corrected = (
        observations
        - fitted_pi[:, None] * p[None, :]
    )

    motion_params = (
        corrected @ linear_inverse.T
    )

    parallax_prediction = (
        motion_params @ linear.T
        + fitted_pi[:, None] * p[None, :]
    )

    parallax_residual = (
        observations - parallax_prediction
    )

    parallax_rms = np.sqrt(
        np.sum(
            parallax_residual ** 2,
            axis=1
        ) / n_epochs
    )

    distance = (
        ARCSEC_PER_RADIAN / fitted_pi
    )

    speed = np.hypot(
        motion_params[:, 2],
        motion_params[:, 3]
    )

    # Approximate angular speed for a
    # circular orbit at this distance.

    circular_speed = (
        1296000 / distance ** 1.5
    )

    return {
        "stationary_rms": stationary_rms,
        "linear_rms": linear_rms,
        "parallax_rms": parallax_rms,
        "raw_parallax": raw_pi,
        "parallax_arcsec": fitted_pi,
        "distance_au": distance,
        "vx_arcsec_year": motion_params[:, 2],
        "vy_arcsec_year": motion_params[:, 3],
        "speed_arcsec_year": speed,
        "circular_speed_ratio":
            speed / circular_speed,
        "parallax_at_boundary":
            (raw_pi <= min_pi)
            | (raw_pi >= max_pi),
        "prediction": parallax_prediction,
        "linear_prediction": linear_prediction,
        "parallax_leverage": denominator
    }
