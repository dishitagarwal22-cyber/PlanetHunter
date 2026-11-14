
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from scipy.ndimage import label

# Load residual map

with fits.open("data/psf_residual.fits") as hdul:
    residual = hdul[0].data
    wcs = WCS(hdul[0].header)

threshold = 4.0
min_pixels = 2

candidates = []

# Detect positive and negative residuals

for sign in [1, -1]:

    signed = sign * residual

    mask = (
        np.isfinite(signed)
        & (signed >= threshold)
    )

    labels, count = label(
        mask,
        structure=np.ones((3, 3))
    )

    for i in range(1, count + 1):

        region = labels == i

        yy, xx = np.where(region)

        if len(xx) < min_pixels:
            continue

        weights = signed[region]

        # Intensity-weighted centroid

        x = np.average(xx, weights=weights)
        y = np.average(yy, weights=weights)

        peak = np.max(weights)

        sky = wcs.pixel_to_world(x, y)

        candidates.append({
            "x": x,
            "y": y,
            "ra": sky.ra.deg,
            "dec": sky.dec.deg,
            "sign": sign,
            "peak": peak,
            "pixels": len(xx)
        })

# Save candidate catalogue

columns = [
    "x", "y", "ra", "dec",
    "sign", "peak", "pixels"
]

table = Table(
    rows=[
        [c[k] for k in columns]
        for c in candidates
    ],
    names=columns
)

table.write(
    "data/candidates.csv",
    overwrite=True
)

print("Total candidates:", len(candidates))

for c in candidates:

    print(
        f"x={c['x']:.2f}, "
        f"y={c['y']:.2f}, "
        f"peak={c['peak']:.2f}, "
        f"sign={c['sign']}"
    )

# Plot detected candidates

fig, ax = plt.subplots(figsize=(8, 8))

im = ax.imshow(
    residual,
    origin="lower",
    cmap="RdBu_r",
    vmin=-5,
    vmax=5
)

for i, c in enumerate(candidates):

    color = "lime" if c["sign"] == 1 else "cyan"

    circle = plt.Circle(
        (c["x"], c["y"]),
        radius=2.5,
        fill=False,
        color=color,
        linewidth=1.5
    )

    ax.add_patch(circle)

    ax.text(
        c["x"] + 3,
        c["y"] + 3,
        str(i + 1),
        color=color,
        fontsize=10
    )

plt.colorbar(im, label="Residual / MAD noise")

ax.set_title("PlanetHunter: Candidate Detections")

plt.show()
