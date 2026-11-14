
from astroquery.ipac.irsa import Irsa
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
from pathlib import Path

RA = 337.8783221
DEC = -5.1891588

coord = SkyCoord(RA, DEC, unit="deg")

print("Searching SPHEREx archive...")

results = Irsa.query_sia(
    pos=(coord, 0.01 * u.deg),
    collection=["spherex_dr1", "spherex_qr3"]
)

print("Total observations:", len(results))
print("\nAvailable columns:")
print(results.colnames)

Path("data").mkdir(exist_ok=True)

results.write(
    "data/observations.csv",
    format="csv",
    overwrite=True
)

print("\nDetector 1 observations:\n")

count = 0

for row in results:

    url = str(row["access_url"])

    if "D1_spx" not in url:
        continue

    count += 1

    print("Candidate:", count)

    if "t_min" in results.colnames:
        mjd = float(row["t_min"])
        date = Time(mjd, format="mjd").iso

        print("Date:", date)
        print("Days from first observation:",
              round(mjd - 61181.506703, 2))

    if "em_min" in results.colnames:
        print("Minimum wavelength:",
              row["em_min"])

    if "em_max" in results.colnames:
        print("Maximum wavelength:",
              row["em_max"])

    print("URL:", url)
    print()

print("Total detector 1 candidates:", count)
