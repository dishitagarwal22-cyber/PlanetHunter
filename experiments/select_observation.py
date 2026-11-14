
from astropy.table import Table
from astropy.time import Time
import numpy as np

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

print("Suitable candidates:", len(candidates))

for separation, mjd, url in candidates[:5]:

    print("\nDate:", Time(mjd, format="mjd").iso)
    print("Separation:", round(separation, 2), "days")
    print("URL:", url)

if candidates:

    best = candidates[0]

    with open("data/selected_url.txt", "w") as f:
        f.write(best[2])

    print("\nSelected observation saved!")
