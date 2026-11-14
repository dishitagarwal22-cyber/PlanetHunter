
import numpy as np
import requests
import time
import shutil

from pathlib import Path
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.time import Time
from reproject import reproject_interp

# Configuration

RA = 337.8783221
DEC = -5.1891588

TARGET_EPOCHS = 6
MIN_SEPARATION_DAYS = 14

folder = Path("data/epochs")
folder.mkdir(parents=True, exist_ok=True)

# Reference image and coordinate grid

with fits.open("data/observation1.fits") as hdul:

    reference = hdul["IMAGE"].data.copy()

    reference_wcs = WCS(
        hdul["IMAGE"].header,
        fobj=hdul
    )

reference_header = reference_wcs.to_header(
    relax=True
)

records = []


def register_observation(path):

    with fits.open(path) as hdul:

        hdu = hdul["IMAGE"]

        image = hdu.data.astype(float)

        wcs = WCS(hdu.header, fobj=hdul)

        date = hdu.header.get(
            "DATE-AVG",
            hdu.header["DATE-OBS"]
        )

        mjd = Time(date).mjd

        detector = hdu.header["DETECTOR"]

        aligned, footprint = reproject_interp(
            (image, wcs),
            reference_wcs,
            shape_out=reference.shape
        )

    aligned[footprint <= 0] = np.nan

    overlap = np.mean(np.isfinite(aligned))

    if overlap < 0.8:
        raise ValueError(
            f"Insufficient overlap: {overlap:.2f}"
        )

    output = folder / (
        path.stem + "_aligned.fits"
    )

    header = reference_header.copy()
    header["DATE-AVG"] = date
    header["MJD-AVG"] = mjd
    header["BUNIT"] = "MJy / sr"

    fits.writeto(
        output,
        aligned,
        header,
        overwrite=True
    )

    records.append({
        "date": date,
        "mjd": mjd,
        "detector": detector,
        "original": str(path),
        "aligned": str(output),
        "overlap": overlap
    })

    print(
        f"Registered: {date} | "
        f"Overlap: {overlap:.3f}"
    )


# Include our existing two observations

for i in [1, 2]:

    source = Path(
        f"data/observation{i}.fits"
    )

    destination = folder / (
        f"existing_{i}.fits"
    )

    shutil.copy2(source, destination)

    register_observation(destination)


# Read archive search results

table = Table.read(
    "data/observations.csv"
)

candidates = []

for row in table:

    url = str(row["access_url"])

    if "D1_spx" not in url:
        continue

    try:
        mjd = float(row["t_min"])
    except (TypeError, ValueError):
        continue

    if not np.isfinite(mjd):
        continue

    candidates.append({
        "url": url,
        "mjd": mjd
    })


# Select observations with diverse dates

def temporal_distance(candidate):

    return min(
        abs(candidate["mjd"] - r["mjd"])
        for r in records
    )


params = {
    "center": f"{RA},{DEC}",
    "size": "0.1"
}

session = requests.Session()

while len(records) < TARGET_EPOCHS:

    if not candidates:
        break

    candidates.sort(
        key=temporal_distance,
        reverse=True
    )

    candidate = candidates.pop(0)

    separation = temporal_distance(candidate)

    if separation < MIN_SEPARATION_DAYS:
        break

    print(
        "\nTrying new observation:",
        Time(candidate["mjd"], format="mjd").iso
    )

    print(
        "Nearest existing epoch:",
        round(separation, 2),
        "days away"
    )

    path = folder / (
        f"epoch_{len(records) + 1}.fits"
    )

    temp = path.with_suffix(".tmp")

    for attempt in range(2):

        try:

            response = session.get(
                candidate["url"],
                params=params,
                timeout=90
            )

            response.raise_for_status()

            temp.write_bytes(response.content)

            with fits.open(temp) as hdul:

                if hdul["IMAGE"].data.ndim != 2:
                    raise ValueError(
                        "Invalid image"
                    )

            temp.replace(path)

            register_observation(path)

            break

        except (
            requests.RequestException,
            OSError,
            ValueError,
            KeyError
        ) as e:

            print(
                f"Attempt {attempt + 1} failed:",
                e
            )

            if temp.exists():
                temp.unlink()

            time.sleep(5)


# Sort observations chronologically

records.sort(
    key=lambda r: r["mjd"]
)

catalogue = Table(rows=records)

catalogue.write(
    "data/epochs.csv",
    overwrite=True
)

print("\nMULTI-EPOCH CATALOGUE")

for i, record in enumerate(records, 1):

    print(
        f"Epoch {i}: "
        f"{record['date']} | "
        f"Overlap={record['overlap']:.3f}"
    )

print("\nTotal epochs:", len(records))
print("Catalogue saved to data/epochs.csv")
