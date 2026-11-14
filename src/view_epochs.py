
import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.io import fits

table = Table.read("data/epochs.csv")

n = len(table)

cols = 3
rows = (n + cols - 1) // cols

fig, axes = plt.subplots(
    rows,
    cols,
    figsize=(12, 4 * rows),
    squeeze=False
)

axes = axes.flatten()

for i, row in enumerate(table):

    image = fits.getdata(
        row["aligned"]
    )

    finite = image[np.isfinite(image)]

    axes[i].imshow(
        image,
        origin="lower",
        cmap="gray",
        vmin=np.percentile(finite, 5),
        vmax=np.percentile(finite, 99)
    )

    axes[i].set_title(
        f"Epoch {i + 1}\n"
        f"{str(row['date'])[:10]}"
    )

for i in range(n, len(axes)):
    axes[i].axis("off")

plt.suptitle(
    "PlanetHunter: Multi-Epoch Observations"
)

plt.tight_layout()
plt.show()
