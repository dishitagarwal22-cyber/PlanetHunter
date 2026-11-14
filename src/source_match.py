
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from scipy.ndimage import label
from scipy.optimize import linear_sum_assignment

# Load registered images

image1 = fits.getdata(
    "data/observation1.fits",
    extname="IMAGE"
).astype(float)

image2 = fits.getdata(
    "data/aligned2.fits"
).astype(float)

# Only compare regions visible in both images

valid = np.isfinite(image1) & np.isfinite(image2)

# Exclude image boundaries

valid[:3, :] = False
valid[-3:, :] = False
valid[:, :3] = False
valid[:, -3:] = False


def detect_sources(image, threshold=3.5):

    background = np.median(image[valid])

    noise = 1.4826 * np.median(
        np.abs(image[valid] - background)
    )

    # Background-subtracted image

    signal = image - background

    mask = valid & (signal > threshold * noise)

    labels, count = label(
        mask,
        structure=np.ones((3, 3))
    )

    sources = []

    for i in range(1, count + 1):

        footprint = labels == i

        yy, xx = np.where(footprint)

        if len(xx) < 2:
            continue

        weights = signal[footprint]

        x = np.average(xx, weights=weights)
        y = np.average(yy, weights=weights)

        sources.append({
            "x": x,
            "y": y,
            "footprint": footprint,
            "area": len(xx),
            "flux": np.sum(weights)
        })

    return sources


sources1 = detect_sources(image1)
sources2 = detect_sources(image2)

print("Epoch 1 sources:", len(sources1))
print("Epoch 2 sources:", len(sources2))


# Your overlap metric

def overlap(a, b):

    intersection = np.sum(
        a["footprint"] & b["footprint"]
    )

    return intersection / min(
        a["area"], b["area"]
    )


def displacement(a, b):

    return np.hypot(
        a["x"] - b["x"],
        a["y"] - b["y"]
    )


# Identify stationary sources

overlap_threshold = 0.8
distance_threshold = 1.0

stationary1 = set()
stationary2 = set()

matches = []

# Build a matrix of possible matches

cost = np.full(
    (len(sources1), len(sources2)),
    1e6
)

for i, a in enumerate(sources1):

    for j, b in enumerate(sources2):

        o = overlap(a, b)
        d = displacement(a, b)

        if (
            o >= overlap_threshold
            and d <= distance_threshold
        ):

            cost[i, j] = (1 - o) + d


# One-to-one matching

if len(sources1) and len(sources2):

    rows, cols = linear_sum_assignment(cost)

    for i, j in zip(rows, cols):

        if cost[i, j] >= 1e6:
            continue

        stationary1.add(i)
        stationary2.add(j)

        matches.append((i, j))

        print(
            f"Stationary match: "
            f"Overlap={overlap(sources1[i], sources2[j]):.2f}, "
            f"Displacement={displacement(sources1[i], sources2[j]):.2f}"
        )


# Remaining sources

remaining1 = [
    s for i, s in enumerate(sources1)
    if i not in stationary1
]

remaining2 = [
    s for i, s in enumerate(sources2)
    if i not in stationary2
]

print("\nStationary matches:", len(matches))
print("Unmatched Epoch 1:", len(remaining1))
print("Unmatched Epoch 2:", len(remaining2))


# Visualize the results

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

for ax, image, sources, stationary, title in zip(

    axes,

    [image1, image2],

    [sources1, sources2],

    [stationary1, stationary2],

    ["Epoch 1", "Epoch 2"]

):

    ax.imshow(
        image,
        origin="lower",
        cmap="gray",
        vmin=np.percentile(image[valid], 5),
        vmax=np.percentile(image[valid], 99)
    )

    for i, source in enumerate(sources):

        color = (
            "lime" if i in stationary
            else "red"
        )

        ax.contour(
            source["footprint"].astype(float),
            levels=[0.5],
            colors=[color],
            linewidths=1.5
        )

        ax.plot(
            source["x"],
            source["y"],
            "+",
            color=color
        )

    ax.set_title(title)

plt.suptitle(
    "Green: Stationary | Red: Unmatched"
)

plt.tight_layout()
plt.show()
