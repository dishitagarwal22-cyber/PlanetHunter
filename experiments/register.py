
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import ZScaleInterval
from reproject import reproject_interp
import matplotlib.pyplot as plt
import numpy as np

with fits.open("data/observation1.fits") as h1:
    image1 = h1["IMAGE"].data.copy()
    wcs1 = WCS(h1["IMAGE"].header, fobj=h1)

with fits.open("data/observation2.fits") as h2:
    image2 = h2["IMAGE"].data.copy()
    wcs2 = WCS(h2["IMAGE"].header, fobj=h2)

aligned2, footprint = reproject_interp(
    (image2, wcs2),
    wcs1,
    shape_out=image1.shape
)

aligned2[footprint == 0] = np.nan

fits.writeto(
    "data/aligned2.fits",
    aligned2,
    wcs1.to_header(relax=True),
    overwrite=True
)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

images = [image1, image2, aligned2]

titles = [
    "Epoch 1",
    "Epoch 2 (Original)",
    "Epoch 2 (Registered)"
]

for ax, image, title in zip(axes, images, titles):

    finite = image[np.isfinite(image)]

    vmin, vmax = ZScaleInterval().get_limits(finite)

    ax.imshow(
        image,
        origin="lower",
        cmap="gray",
        vmin=vmin,
        vmax=vmax
    )

    ax.set_title(title)

plt.tight_layout()
plt.show()

print("Registration complete!")
print("Overlap fraction:", np.mean(footprint > 0))
