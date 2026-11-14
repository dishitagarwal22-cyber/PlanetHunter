
import hashlib
import re
import time

from pathlib import Path

import numpy as np
import pandas as pd
import requests

from astroquery.ipac.irsa import Irsa

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

import astropy.units as u

from reproject import reproject_interp


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

CACHE = ROOT / "data" / "explorer"

CACHE.mkdir(
    parents=True,
    exist_ok=True
)

COLLECTIONS = [
    "spherex_dr1",
    "spherex_qr3"
]

PIXEL_SCALE_ARCSEC = 6.1


# --------------------------------------------------
# COORDINATE VALIDATION
# --------------------------------------------------

def validate_coordinates(ra, dec, fov):

    ra = float(ra)
    dec = float(dec)
    fov = float(fov)

    if not np.all(np.isfinite([ra, dec, fov])):

        raise ValueError(
            "Coordinates and FOV must be finite."
        )

    if not 0 <= ra < 360:

        raise ValueError(
            "RA must be between 0 and 360 degrees."
        )

    if not -90 <= dec <= 90:

        raise ValueError(
            "Dec must be between -90 and 90 degrees."
        )

    if not 0.02 <= fov <= 0.5:

        raise ValueError(
            "FOV must be between 0.02 and 0.5 degrees."
        )

    return ra, dec, fov


# --------------------------------------------------
# DETECTOR IDENTIFICATION
# --------------------------------------------------

def get_detector(url):

    match = re.search(
        r"D([1-6])_spx",
        str(url)
    )

    if match:

        return int(match.group(1))

    return None


# --------------------------------------------------
# SEARCH SPHEREX ARCHIVE
# --------------------------------------------------

def search_spherex(
    ra,
    dec,
    fov=0.1
):

    ra, dec, fov = validate_coordinates(
        ra, dec, fov
    )

    coord = SkyCoord(
        ra,
        dec,
        unit="deg"
    )

    print(
        f"Searching SPHEREx: "
        f"RA={ra}, Dec={dec}"
    )

    results = Irsa.query_sia(

        pos=(
            coord,
            0.01 * u.deg
        ),

        collection=COLLECTIONS
    )

    if len(results) == 0:

        raise RuntimeError(
            "No SPHEREx observations found."
        )

    records = []

    for row in results:

        url = str(
            row["access_url"]
        )

        detector = get_detector(url)

        if detector is None:

            continue

        try:

            mjd = float(
                row["t_min"]
            )

        except (
            TypeError,
            ValueError,
            KeyError
        ):

            continue

        if not np.isfinite(mjd):

            continue

        record = {

            "url": url,

            "detector": detector,

            "mjd": mjd,

            "date": Time(
                mjd,
                format="mjd"
            ).iso
        }

        # SIA wavelength metadata is normally
        # expressed in metres.

        for column in [
            "em_min",
            "em_max"
        ]:

            try:

                value = float(
                    row[column]
                )

                record[column + "_um"] = (
                    value * 1e6
                )

            except (
                TypeError,
                ValueError,
                KeyError
            ):

                record[column + "_um"] = (
                    np.nan
                )

        records.append(record)

    if not records:

        raise RuntimeError(
            "No usable detector observations found."
        )

    catalogue = pd.DataFrame(
        records
    )

    catalogue = catalogue.drop_duplicates(

        subset=["url"]
    )

    catalogue = catalogue.sort_values(

        "mjd"

    ).reset_index(drop=True)

    print(
        "Usable observations:",
        len(catalogue)
    )

    return catalogue


# --------------------------------------------------
# CONSTRUCT COMMON SKY GRID
# --------------------------------------------------

def make_reference_grid(
    ra,
    dec,
    fov
):

    scale = (
        PIXEL_SCALE_ARCSEC / 3600
    )

    n_pixels = int(
        np.ceil(fov / scale)
    )

    # Odd dimensions place the requested
    # sky coordinate at the central pixel.

    if n_pixels % 2 == 0:

        n_pixels += 1

    wcs = WCS(naxis=2)

    wcs.wcs.ctype = [

        "RA---TAN",
        "DEC--TAN"
    ]

    wcs.wcs.cunit = [

        "deg",
        "deg"
    ]

    wcs.wcs.crval = [

        ra,
        dec
    ]

    wcs.wcs.crpix = [

        (n_pixels + 1) / 2,

        (n_pixels + 1) / 2
    ]

    wcs.wcs.cdelt = [

        -scale,

        scale
    ]

    shape = (
        n_pixels,
        n_pixels
    )

    return wcs, shape


# --------------------------------------------------
# CACHE DIRECTORY
# --------------------------------------------------

def get_cache_folder(
    ra,
    dec,
    fov
):

    key = (

        f"{ra:.7f}_"

        f"{dec:.7f}_"

        f"{fov:.5f}"
    )

    folder = CACHE / key

    folder.mkdir(

        parents=True,

        exist_ok=True
    )

    return folder


# --------------------------------------------------
# DOWNLOAD FITS CUTOUT
# --------------------------------------------------

def download_cutout(
    url,
    ra,
    dec,
    fov,
    folder,
    session
):

    key = hashlib.sha256(

        url.encode()

    ).hexdigest()[:16]

    path = folder / (

        f"{key}.fits"
    )

    if path.exists():

        try:

            with fits.open(path) as hdul:

                if hdul["IMAGE"].data.ndim == 2:

                    return path

        except (
            OSError,
            ValueError,
            KeyError
        ):

            pass

        path.unlink(missing_ok=True)

    params = {

        "center": f"{ra},{dec}",

        "size": str(fov)
    }

    temp = path.with_suffix(".tmp")

    last_error = None

    for attempt in range(2):

        try:

            response = session.get(

                url,

                params=params,

                timeout=90
            )

            response.raise_for_status()

            temp.write_bytes(
                response.content
            )

            with fits.open(temp) as hdul:

                image = hdul["IMAGE"].data

                if image.ndim != 2:

                    raise ValueError(
                        "Invalid FITS image."
                    )

            temp.replace(path)

            return path

        except (
            requests.RequestException,
            OSError,
            ValueError,
            KeyError
        ) as error:

            last_error = error

            temp.unlink(missing_ok=True)

            print(

                f"Download attempt "
                f"{attempt + 1} failed:",

                error
            )

            if attempt == 0:

                time.sleep(3)

    raise RuntimeError(

        f"Could not download FITS: "
        f"{last_error}"
    )


# --------------------------------------------------
# REGISTER ONE OBSERVATION
# --------------------------------------------------

def register_cutout(
    path,
    reference_wcs,
    reference_shape,
    min_overlap=0.7
):

    with fits.open(path) as hdul:

        hdu = hdul["IMAGE"]

        image = hdu.data.astype(float)

        source_wcs = WCS(

            hdu.header,

            fobj=hdul

        ).celestial

        date = (

            hdu.header.get("DATE-AVG")

            or hdu.header.get("DATE-OBS")
        )

        if date is None:

            raise ValueError(
                "Observation date missing."
            )

        mjd = Time(date).mjd

        detector = hdu.header.get(

            "DETECTOR",

            -1
        )

        bunit = hdu.header.get(

            "BUNIT",

            ""
        )

        aligned, footprint = reproject_interp(

            (image, source_wcs),

            reference_wcs,

            shape_out=reference_shape
        )

    aligned[footprint <= 0] = np.nan

    overlap = np.mean(

        np.isfinite(aligned)
    )

    if overlap < min_overlap:

        raise ValueError(

            f"Insufficient image overlap: "
            f"{overlap:.3f}"
        )

    output = path.with_name(

        path.stem + "_aligned.fits"
    )

    header = reference_wcs.to_header(

        relax=True
    )

    header["DATE-AVG"] = date

    header["MJD-AVG"] = mjd

    header["DETECTOR"] = detector

    header["BUNIT"] = bunit

    fits.writeto(

        output,

        aligned,

        header,

        overwrite=True
    )

    return {

        "date": date,

        "mjd": mjd,

        "detector": detector,

        "original": str(path),

        "aligned": str(output),

        "overlap": float(overlap),

        "bunit": bunit
    }


# --------------------------------------------------
# BUILD MULTI-EPOCH IMAGE SEQUENCE
# --------------------------------------------------

def build_sequence(
    ra,
    dec,
    fov=0.1,
    detector=1,
    target_epochs=6,
    min_separation_days=14,
    catalogue=None
):

    ra, dec, fov = validate_coordinates(

        ra, dec, fov
    )

    if detector not in range(1, 7):

        raise ValueError(
            "Detector must be between 1 and 6."
        )

    if catalogue is None:

        catalogue = search_spherex(

            ra,

            dec,

            fov
        )

    candidates = catalogue[

        catalogue["detector"] == detector

    ].copy()

    if candidates.empty:

        raise RuntimeError(

            f"No observations available "
            f"for detector {detector}."
        )

    candidates = candidates.sort_values(

        "mjd"

    ).to_dict("records")

    reference_wcs, reference_shape = (

        make_reference_grid(

            ra,

            dec,

            fov
        )
    )

    folder = get_cache_folder(

        ra,

        dec,

        fov
    )

    records = []

    errors = []

    def temporal_distance(candidate):

        return min(

            abs(

                candidate["mjd"]

                - record["mjd"]
            )

            for record in records
        )

    with requests.Session() as session:

        while (

            len(records) < target_epochs

            and candidates
        ):

            if records:

                candidates.sort(

                    key=temporal_distance,

                    reverse=True
                )

            candidate = candidates.pop(0)

            if records:

                separation = temporal_distance(

                    candidate
                )

                if separation < min_separation_days:

                    break

            print(

                "\nTrying observation:",

                candidate["date"]
            )

            try:

                path = download_cutout(

                    candidate["url"],

                    ra,

                    dec,

                    fov,

                    folder,

                    session
                )

                record = register_cutout(

                    path,

                    reference_wcs,

                    reference_shape
                )

                record["archive_url"] = (

                    candidate["url"]
                )

                record["em_min_um"] = (

                    candidate["em_min_um"]
                )

                record["em_max_um"] = (

                    candidate["em_max_um"]
                )

                records.append(record)

                print(

                    "Registered:",

                    record["date"],

                    "| Overlap:",

                    round(
                        record["overlap"],
                        3
                    )
                )

            except Exception as error:

                errors.append({

                    "date":
                        candidate["date"],

                    "error":
                        str(error)
                })

                print(

                    "Observation skipped:",

                    error
                )

    if not records:

        raise RuntimeError(

            "No observations could be "
            "downloaded and registered."
        )

    sequence = pd.DataFrame(

        records

    ).sort_values(

        "mjd"

    ).reset_index(drop=True)

    sequence.insert(

        0,

        "epoch_id",

        np.arange(
            1,
            len(sequence) + 1
        )
    )

    output = folder / (

        f"detector_{detector}_epochs.csv"
    )

    sequence.to_csv(

        output,

        index=False
    )

    if errors:

        pd.DataFrame(errors).to_csv(

            folder / (
                f"detector_{detector}_errors.csv"
            ),

            index=False
        )

    print(

        "\nTotal registered epochs:",

        len(sequence)
    )

    print(

        "Sequence saved:",

        output
    )

    return sequence


# --------------------------------------------------
# LOAD ALIGNED IMAGE STACK
# --------------------------------------------------

def load_image_stack(sequence):

    images = []

    for path in sequence["aligned"]:

        with fits.open(path) as hdul:

            images.append(

                hdul[0].data.astype(float)
            )

    return np.stack(images)


# --------------------------------------------------
# COMMON DISPLAY LIMITS
# --------------------------------------------------

def get_display_limits(
    images,
    lower=5,
    upper=99.5
):

    values = images[

        np.isfinite(images)
    ]

    if len(values) == 0:

        raise ValueError(
            "No finite image pixels."
        )

    vmin, vmax = np.percentile(

        values,

        [lower, upper]
    )

    if vmax <= vmin:

        vmax = vmin + 1e-6

    return float(vmin), float(vmax)
