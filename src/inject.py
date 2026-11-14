
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Load original observations

image1 = fits.getdata(
    "data/observation1.fits",
    extname="IMAGE"
).astype(float)

image2 = fits.getdata(
    "data/aligned2.fits"
).astype(float)


def inject_source(image, x, y, amplitude, sigma):

    yy, xx = np.indices(image.shape)

    source = amplitude * np.exp(
        -((xx - x)**2 + (yy - y)**2)
        / (2 * sigma**2)
    )

    return image + source


# Epoch 1 is May 2026

injected1 = inject_source(
    image1,
    x=35,
    y=30,
    amplitude=0.5,
    sigma=0.9
)

# Epoch 2 is May 2025

injected2 = inject_source(
    image2,
    x=15,
    y=20,
    amplitude=0.5,
    sigma=1.2
)

# Save separate files
# Preserve original observations

header = fits.getheader("data/aligned2.fits")

fits.writeto(
    "data/injected1.fits",
    injected1,
    header,
    overwrite=True
)

fits.writeto(
    "data/injected2.fits",
    injected2,
    header,
    overwrite=True
)

print("Synthetic moving source injected!")
print("True displacement:", np.hypot(20, 10))

# Display injected sources

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

for ax, image, x, y, title in [
    (axes[0], injected2, 15, 20, "May 2025"),
    (axes[1], injected1, 35, 30, "May 2026")
]:

    finite = image[np.isfinite(image)]

    ax.imshow(
        image,
        origin="lower",
        cmap="gray",
        vmin=np.percentile(finite, 5),
        vmax=np.percentile(finite, 99)
    )

    ax.add_patch(
        plt.Circle(
            (x, y),
            radius=3,
            fill=False,
            color="yellow",
            linewidth=2
        )
    )

    ax.set_title(title)

plt.suptitle("Synthetic Moving Object Test")
plt.tight_layout()
plt.show()
