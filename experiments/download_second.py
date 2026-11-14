
import requests
import time
import numpy as np

from pathlib import Path
from astropy.table import Table
from astropy.io import fits

table = Table.read("data/observations.csv")

reference_mjd = 61181.506703
candidates = []

for row in table:

    url = str(row["access_url"])

    if "D1_spx" not in url:
        continue

    mjd = float(row["t_min"])

    if not np.isfinite(mjd):
        continue

    separation = abs(mjd - reference_mjd)

    if separation < 30:
        continue

    candidates.append((separation, mjd, url))

candidates.sort(reverse=True)

params = {
    "center": "337.8783221,-5.1891588",
    "size": "0.1"
}

success = False

for separation, mjd, url in candidates[:5]:

    print("\nTrying observation:", url.split("/")[-1])
    print("Epoch separation:", round(separation, 2), "days")

    for attempt in range(3):

        try:

            response = requests.get(
                url,
                params=params,
                timeout=60
            )

            response.raise_for_status()

            path = Path("data/observation2.fits")
            temp = Path("data/observation2.tmp")

            temp.write_bytes(response.content)

            with fits.open(temp) as hdul:
                image = hdul["IMAGE"].data

                if image is None or image.ndim != 2:
                    raise ValueError("Invalid science image")

                print("Date:", hdul["IMAGE"].header["DATE-OBS"])
                print("Image shape:", image.shape)

            temp.replace(path)

            print("Download successful!")
            success = True
            break

        except (requests.RequestException, OSError, ValueError, KeyError) as e:

            print("Attempt", attempt + 1, "failed:", e)

            if attempt < 2:
                time.sleep(5 * (attempt + 1))

    if success:
        break

if not success:
    print("\nAll candidate downloads failed.")
    print("IRSA may be temporarily unavailable.")
