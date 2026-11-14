
from astropy.io import fits
from astropy.visualization import ZScaleInterval
import matplotlib.pyplot as plt
import numpy as np

with fits.open("data/observation1.fits") as hdul:

    for hdu in hdul:

        if hdu.data is not None:
            if hdu.data.ndim == 2:
                image = hdu.data
                print("Selected HDU:", hdu.name)
                break

    else:
        raise ValueError("No 2D image found")

    print("Image shape:", image.shape)

    finite_pixels = image[np.isfinite(image)]

    vmin, vmax = ZScaleInterval().get_limits(finite_pixels)

    plt.figure(figsize=(8, 8))

    plt.imshow(
        image,
        origin="lower",
        cmap="gray",
        vmin=vmin,
        vmax=vmax
    )

    plt.colorbar(label="Pixel intensity")
    plt.title("SPHEREx Observation")
    plt.xlabel("X pixel")
    plt.ylabel("Y pixel")
    plt.show()

from astropy.io import fits
from astropy.visualization import ZScaleInterval
import matplotlib.pyplot as plt
import numpy as np

with fits.open("data/observation1.fits") as hdul:

    image = hdul["IMAGE"].data
    variance = hdul["VARIANCE"].data
    flags = hdul["FLAGS"].data

    print("Image shape:", image.shape)
    print("Minimum:", np.nanmin(image))
    print("Maximum:", np.nanmax(image))
    print("Median:", np.nanmedian(image))

    finite = image[np.isfinite(image)]
    vmin, vmax = ZScaleInterval().get_limits(finite)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(
        image,
        origin="lower",
        cmap="gray",
        vmin=vmin,
        vmax=vmax
    )
    axes[0].set_title("Science Image")

    axes[1].imshow(
        np.sqrt(np.maximum(variance, 0)),
        origin="lower",
        cmap="inferno"
    )
    axes[1].set_title("Pixel Uncertainty")

    axes[2].imshow(
        flags != 0,
        origin="lower",
        cmap="gray"
    )
    axes[2].set_title("Nonzero Quality Flags")

    plt.tight_layout()
    plt.show()
