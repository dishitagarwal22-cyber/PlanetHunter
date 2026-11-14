
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares

image1 = fits.getdata(
    "data/observation1.fits", extname="IMAGE"
).astype(float)

image2 = fits.getdata(
    "data/aligned2.fits"
).astype(float)

# Replace missing pixels for convolution

filled1 = np.where(
    np.isfinite(image1),
    image1,
    np.nanmedian(image1)
)

filled2 = np.where(
    np.isfinite(image2),
    image2,
    np.nanmedian(image2)
)

# Exclude invalid pixels and image boundaries

valid = np.isfinite(image1) & np.isfinite(image2)

valid[:6, :] = False
valid[-6:, :] = False
valid[:, :6] = False
valid[:, -6:] = False

# Fit both possible smoothing directions

def fit_direction(source, target):

    def residual(params):

        a, b, sigma = params

        blurred = gaussian_filter(source, sigma)

        model = a * blurred + b

        return (target - model)[valid]

    initial = [
        1.0,
        np.median(target[valid])
        - np.median(source[valid]),
        0.7
    ]

    r = residual(initial)

    noise = 1.4826 * np.median(
        np.abs(r - np.median(r))
    )

    fit = least_squares(
        residual,
        initial,
        bounds=(
            [0, -np.inf, 0],
            [np.inf, np.inf, 2]
        ),
        loss="soft_l1",
        f_scale=max(noise, 1e-8)
    )

    return fit

fit1 = fit_direction(filled1, filled2)
fit2 = fit_direction(filled2, filled1)

# Construct matched images in epoch 2 units

if fit1.cost <= fit2.cost:

    a, b, sigma = fit1.x

    matched1 = a * gaussian_filter(
        filled1, sigma
    ) + b

    matched2 = filled2

    direction = "Epoch 1 smoothed"

else:

    a, b, sigma = fit2.x

    if a <= 1e-8:
        raise ValueError("Invalid photometric scale")

    matched1 = (filled1 - b) / a

    matched2 = gaussian_filter(
        filled2, sigma
    )

    direction = "Epoch 2 smoothed"

# Difference image

difference = matched2 - matched1

difference[~valid] = np.nan

median = np.nanmedian(difference)

noise = 1.4826 * np.nanmedian(
    np.abs(difference - median)
)

if noise <= 0:
    raise ValueError("Invalid noise estimate")

residual_map = (difference - median) / noise

print("Direction:", direction)
print("Gaussian sigma:", round(sigma, 3))
print("Photometric scale:", round(a, 4))
print("Background offset:", round(b, 4))
print("Residual noise:", round(noise, 6))

# Save results with spatial WCS

header = fits.getheader("data/aligned2.fits")

fits.writeto(
    "data/psf_difference.fits",
    difference,
    header,
    overwrite=True
)

fits.writeto(
    "data/psf_residual.fits",
    residual_map,
    header,
    overwrite=True
)

# Compare previous and improved residuals

old = fits.getdata("data/residual_map.fits")

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

for ax, image, title in zip(
    axes,
    [old, residual_map],
    ["Before PSF Matching", "After PSF Matching"]
):

    im = ax.imshow(
        image,
        origin="lower",
        cmap="RdBu_r",
        vmin=-5,
        vmax=5
    )

    ax.set_title(title)

fig.colorbar(
    im,
    ax=axes,
    label="Residual / MAD noise"
)

plt.show()
