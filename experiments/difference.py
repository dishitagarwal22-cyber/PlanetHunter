
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.visualization import ZScaleInterval
from scipy.optimize import least_squares

# Load both registered observations

with fits.open("data/observation1.fits") as hdul:
    image1 = hdul["IMAGE"].data.astype(float)

image2 = fits.getdata("data/aligned2.fits").astype(float)

# Identify overlapping valid pixels

valid = np.isfinite(image1) & np.isfinite(image2)

x = image1[valid]
y = image2[valid]

# Estimate photometric scaling and background offset

def residual(params):
    a, b = params
    return y - (a * x + b)

initial = [1.0, np.median(y) - np.median(x)]

initial_residual = residual(initial)

noise = 1.4826 * np.median(
    np.abs(initial_residual - np.median(initial_residual))
)

fit = least_squares(
    residual,
    initial,
    loss="soft_l1",
    f_scale=max(noise, 1e-8),
    bounds=([0, -np.inf], [np.inf, np.inf])
)

a, b = fit.x

print("Photometric scale:", a)
print("Background offset:", b)

# Normalize first observation

normalized1 = a * image1 + b

# Calculate difference image

difference = image2 - normalized1

difference[~valid] = np.nan

# Estimate robust residual noise

median = np.nanmedian(difference)

mad = np.nanmedian(
    np.abs(difference - median)
)

sigma = 1.4826 * mad

if sigma <= 0:
    raise ValueError("Residual noise estimate is zero")

residual_map = (difference - median) / sigma

print("Residual noise:", sigma)

# Save results

fits.writeto(
    "data/difference.fits",
    difference,
    overwrite=True
)

fits.writeto(
    "data/residual_map.fits",
    residual_map,
    overwrite=True
)

# Visualize

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, image, title in zip(
    axes[:2],
    [normalized1, image2],
    ["Normalized Epoch 1", "Registered Epoch 2"]
):
    ax.imshow(
        image,
        origin="lower",
        cmap="gray",
        vmin=np.percentile(y, 5),
        vmax=np.percentile(y, 99)
    )
    ax.set_title(title)

im = axes[2].imshow(
    residual_map,
    origin="lower",
    cmap="RdBu_r",
    vmin=-5,
    vmax=5
)

axes[2].set_title("Normalized Residual Map")

plt.colorbar(im, ax=axes[2], label="Residual / MAD noise")

plt.tight_layout()
plt.show()
