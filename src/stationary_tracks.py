
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
import astropy.units as u


CATALOGUE = "data/source_catalogue.csv"
PAIRS = "data/stationary_pairs.csv"

OUTPUT = "data/stationary_tracks.csv"
MEMBERS_OUTPUT = "data/track_members.csv"

MAX_SEPARATION = 6.1


# Union-Find data structure

class UnionFind:

    def __init__(self, elements):

        self.parent = {
            element: element
            for element in elements
        }

    def find(self, x):

        if self.parent[x] != x:

            self.parent[x] = self.find(
                self.parent[x]
            )

        return self.parent[x]

    def union(self, a, b):

        root_a = self.find(a)
        root_b = self.find(b)

        if root_a != root_b:

            self.parent[root_b] = root_a


def main():

    catalogue = pd.read_csv(CATALOGUE)

    pairs = pd.read_csv(PAIRS)

    # Initially every detection is its own group

    uf = UnionFind(
        catalogue["source_id"]
    )

    # Connect mutual nearest associations

    for _, pair in pairs.iterrows():

        if (
            pair["mutual_nearest"]
            and pair["separation_arcsec"]
            <= MAX_SEPARATION
        ):

            uf.union(
                pair["source1"],
                pair["source2"]
            )

    # Construct connected groups

    groups = {}

    for source_id in catalogue["source_id"]:

        root = uf.find(source_id)

        if root not in groups:

            groups[root] = []

        groups[root].append(source_id)

    # Sort groups by number of detections

    ordered_groups = sorted(
        groups.values(),
        key=lambda g: (-len(g), min(g))
    )

    tracks = []
    members = []

    print("STATIONARY TRACK ANALYSIS")

    for index, group in enumerate(
        ordered_groups, 1
    ):

        track_id = f"S{index:03d}"

        detections = catalogue[
            catalogue["source_id"].isin(group)
        ].sort_values("mjd")

        n_detections = len(detections)

        n_epochs = detections[
            "epoch_id"
        ].nunique()

        # Check for conflicting detections
        # within the same epoch

        duplicate_epoch = (
            n_epochs != n_detections
        )

        coords = SkyCoord(
            ra=detections["ra"].to_numpy() * u.deg,
            dec=detections["dec"].to_numpy() * u.deg
        )

        # Calculate all pairwise separations

        separations = coords[:, None].separation(
            coords[None, :]
        ).arcsec

        max_separation = float(
            np.max(separations)
        )

        # Local tangent-plane offsets
        # relative to first detection

        offset_ra, offset_dec = (
            coords[0].spherical_offsets_to(coords)
        )

        dx = offset_ra.arcsec
        dy = offset_dec.arcsec

        # Scatter around mean position

        rms_scatter = np.sqrt(
            np.mean(
                (dx - np.mean(dx)) ** 2
                +
                (dy - np.mean(dy)) ** 2
            )
        )

        # Classify the group

        if duplicate_epoch:

            status = "conflicting"

        elif n_detections == 1:

            status = "unassociated"

        elif max_separation > MAX_SEPARATION:

            status = "inconsistent"

        elif n_epochs >= 3:

            status = "stationary_candidate"

        else:

            status = "two_epoch_candidate"

        record = {
            "track_id": track_id,
            "n_detections": n_detections,
            "n_epochs": n_epochs,
            "mean_x": detections["x"].mean(),
            "mean_y": detections["y"].mean(),
            "max_separation_arcsec": max_separation,
            "rms_scatter_arcsec": rms_scatter,
            "duplicate_epoch": duplicate_epoch,
            "status": status,
            "source_ids": ";".join(
                detections["source_id"]
            )
        }

        tracks.append(record)

        # Save individual track members

        for _, detection in detections.iterrows():

            members.append({
                "track_id": track_id,
                "source_id": detection["source_id"],
                "epoch_id": detection["epoch_id"],
                "mjd": detection["mjd"],
                "ra": detection["ra"],
                "dec": detection["dec"],
                "x": detection["x"],
                "y": detection["y"],
                "peak_snr": detection["peak_snr"],
                "status": status
            })

        print(
            f"\n{track_id} | "
            f"{n_epochs} epochs | "
            f"{status}"
        )

        print(
            "Members:",
            record["source_ids"]
        )

        print(
            f"Maximum separation: "
            f"{max_separation:.3f} arcsec"
        )

        print(
            f"RMS scatter: "
            f"{rms_scatter:.3f} arcsec"
        )

    tracks_df = pd.DataFrame(tracks)

    members_df = pd.DataFrame(members)

    tracks_df.to_csv(
        OUTPUT,
        index=False
    )

    members_df.to_csv(
        MEMBERS_OUTPUT,
        index=False
    )

    print("\nSUMMARY")

    print(
        "Total detections:",
        len(catalogue)
    )

    print(
        "Total source groups:",
        len(tracks_df)
    )

    print("\nGroups by number of epochs:")

    print(
        tracks_df["n_epochs"]
        .value_counts()
        .sort_index()
    )

    print("\nGroups by status:")

    print(
        tracks_df["status"].value_counts()
    )

    # Visualize all source groups

    fig, ax = plt.subplots(
        figsize=(10, 10)
    )

    colors = {
        "stationary_candidate": "green",
        "two_epoch_candidate": "orange",
        "unassociated": "red",
        "inconsistent": "purple",
        "conflicting": "blue"
    }

    for _, track in tracks_df.iterrows():

        detections = members_df[
            members_df["track_id"]
            == track["track_id"]
        ].sort_values("mjd")

        color = colors[track["status"]]

        ax.plot(
            detections["x"],
            detections["y"],
            "o-",
            color=color,
            markersize=4,
            alpha=0.8
        )

        ax.annotate(
            track["track_id"],
            (
                track["mean_x"],
                track["mean_y"]
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color=color
        )

    ax.set_xlabel("X pixel")
    ax.set_ylabel("Y pixel")

    ax.set_title(
        "PlanetHunter: Stationary Source Groups"
    )

    ax.set_aspect("equal")
    ax.grid(alpha=0.2)

    # Legend

    for status, color in colors.items():

        ax.plot(
            [],
            [],
            "o",
            color=color,
            label=status
        )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        "data/stationary_tracks.png",
        dpi=150
    )

    plt.show()


if __name__ == "__main__":
    main()
