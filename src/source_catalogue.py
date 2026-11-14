
import csv
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS

from source_detection import detect_sources


# Configuration

THRESHOLD = 3.5
MIN_AREA = 2

EPOCH_CATALOGUE = Path("data/epochs.csv")
OUTPUT_CATALOGUE = Path("data/source_catalogue.csv")
OUTPUT_FOOTPRINTS = Path("data/source_footprints.npz")


def main():

    # Read the epoch catalogue

    with EPOCH_CATALOGUE.open() as file:
        epochs = list(csv.DictReader(file))

    epochs.sort(key=lambda e: float(e["mjd"]))

    if not epochs:
        raise ValueError("No epochs found")

    records = []
    footprints = {}

    # Prepare visualization

    ncols = 2
    nrows = (len(epochs) + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(12, 6 * nrows),
        squeeze=False
    )

    axes = axes.ravel()

    # Process each observation

    for epoch_id, epoch in enumerate(epochs, 1):

        path = epoch["aligned"]

        with fits.open(path) as hdul:

            image = hdul[0].data.astype(float)

            wcs = WCS(
                hdul[0].header
            ).celestial

        sources = detect_sources(
            image,
            threshold=THRESHOLD,
            min_area=MIN_AREA
        )

        print(
            f"\nEpoch {epoch_id}: "
            f"{epoch['date']}"
        )

        print(
            f"Detected sources: {len(sources)}"
        )

        # Display the image

        ax = axes[epoch_id - 1]

        valid = np.isfinite(image)

        ax.imshow(
            image,
            origin="lower",
            cmap="gray",
            vmin=np.percentile(image[valid], 5),
            vmax=np.percentile(image[valid], 99)
        )

        # Process each detected source

        for index, source in enumerate(sources, 1):

            source_id = (
                f"E{epoch_id}_{index:03d}"
            )

            x = source["x"]
            y = source["y"]

            # Pixel coordinates -> sky coordinates

            ra, dec = wcs.pixel_to_world_values(
                x, y
            )

            record = {
                "epoch_id": epoch_id,
                "source_id": source_id,
                "date": epoch["date"],
                "mjd": float(epoch["mjd"]),
                "detector": epoch["detector"],
                "original_file": epoch["original"],
                "aligned_file": path,
                "x": x,
                "y": y,
                "ra": float(ra),
                "dec": float(dec),
                "area": source["area"],
                "peak": source["peak"],
                "integrated_signal":
                    source["integrated_signal"],
                "peak_snr": source["peak_snr"],
                "sigma_x": source["sigma_x"],
                "sigma_y": source["sigma_y"],
                "background": source["background"],
                "noise": source["noise"]
            }

            records.append(record)

            # Preserve the complete source footprint

            footprints[source_id] = (
                source["footprint"]
            )

            # Plot centroid and footprint

            ax.contour(
                source["footprint"].astype(float),
                levels=[0.5],
                colors=["cyan"],
                linewidths=1
            )

            ax.plot(
                x,
                y,
                "+",
                color="yellow",
                markersize=7
            )

            ax.annotate(
                str(index),
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                color="yellow",
                fontsize=8
            )

        ax.set_title(
            f"Epoch {epoch_id} | "
            f"{epoch['date'][:10]}\n"
            f"{len(sources)} sources"
        )

        ax.set_xlabel("X pixel")
        ax.set_ylabel("Y pixel")

    # Hide unused panels

    for ax in axes[len(epochs):]:
        ax.set_visible(False)

    # Save the source catalogue

    if not records:
        raise ValueError(
            "No sources detected in any epoch"
        )

    with OUTPUT_CATALOGUE.open(
        "w", newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0].keys())
        )

        writer.writeheader()
        writer.writerows(records)

    # Save footprints in compressed NumPy format

    np.savez_compressed(
        OUTPUT_FOOTPRINTS,
        **footprints
    )

    print("\nCATALOGUE SUMMARY")

    print("Total epochs:", len(epochs))
    print("Total detections:", len(records))

    print(
        "Catalogue saved to:",
        OUTPUT_CATALOGUE
    )

    print(
        "Footprints saved to:",
        OUTPUT_FOOTPRINTS
    )

    plt.suptitle(
        "PlanetHunter: Multi-Epoch Source Catalogue"
    )

    plt.tight_layout()

    plt.savefig(
        "data/source_catalogue_preview.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()


if __name__ == "__main__":
    main()
