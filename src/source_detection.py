
import numpy as np
from scipy.ndimage import label


def detect_sources(
    image,
    threshold=3.5,
    min_area=2,
    border=3,
    valid_mask=None
):

    image = np.asarray(image, dtype=float)

    # Each image has its own validity mask
    valid = np.isfinite(image)

    if valid_mask is not None:
        valid &= valid_mask

    # Exclude image boundaries
    if border > 0:
        valid[:border, :] = False
        valid[-border:, :] = False
        valid[:, :border] = False
        valid[:, -border:] = False

    if not np.any(valid):
        raise ValueError("No valid pixels in image")

    # Estimate background and noise
    background = np.median(image[valid])

    noise = 1.4826 * np.median(
        np.abs(image[valid] - background)
    )

    if not np.isfinite(noise) or noise <= 0:
        raise ValueError("Invalid noise estimate")

    # Background subtraction
    signal = image - background

    # Identify significant pixels
    mask = valid & (signal > threshold * noise)

    labels, count = label(
        mask,
        structure=np.ones((3, 3))
    )

    sources = []

    for i in range(1, count + 1):

        footprint = labels == i

        yy, xx = np.where(footprint)

        area = len(xx)

        if area < min_area:
            continue

        weights = signal[footprint]

        # Intensity-weighted centroid
        x = np.average(xx, weights=weights)
        y = np.average(yy, weights=weights)

        # Source measurements
        peak = np.max(weights)

        integrated_signal = np.sum(weights)

        peak_snr = peak / noise

        # Intensity-weighted spatial spread
        sigma_x = np.sqrt(
            np.average(
                (xx - x) ** 2,
                weights=weights
            )
        )

        sigma_y = np.sqrt(
            np.average(
                (yy - y) ** 2,
                weights=weights
            )
        )

        sources.append({
            "x": float(x),
            "y": float(y),
            "area": int(area),
            "peak": float(peak),
            "integrated_signal": float(
                integrated_signal
            ),
            "peak_snr": float(peak_snr),
            "sigma_x": float(sigma_x),
            "sigma_y": float(sigma_y),
            "background": float(background),
            "noise": float(noise),
            "footprint": footprint
        })

    return sources
