
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
import astropy.units as u

from sky_explorer import (

    search_spherex,

    build_sequence,

    load_image_stack,

    get_display_limits
)


# Approximate J2000 coordinates
# of Barnard's Star.

coord = SkyCoord(

    "17h57m48.5s",

    "+04d41m36s",

    frame="icrs"
)

RA = coord.ra.deg

DEC = coord.dec.deg

FOV = 0.1


print("\nTARGET: BARNARD'S STAR")

print("RA:", RA)

print("Dec:", DEC)


# --------------------------------------------------
# SEARCH ARCHIVE
# --------------------------------------------------

catalogue = search_spherex(

    RA,

    DEC,

    FOV
)

print("\nAVAILABLE DETECTORS")

print(

    catalogue.groupby("detector")

    .size()

    .to_string()
)

print("\nDATE COVERAGE")

print(

    catalogue.groupby("detector")

    .agg(

        observations=("mjd", "count"),

        first_date=("date", "min"),

        last_date=("date", "max")
    )

    .to_string()
)


# --------------------------------------------------
# SELECT DETECTOR
# --------------------------------------------------

# Detector 1 is our initial choice
# because we have already worked with it.

if 1 in catalogue["detector"].values:

    detector = 1

else:

    detector = int(

        catalogue["detector"]

        .value_counts()

        .index[0]
    )

print(

    "\nSelected detector:",

    detector
)


# --------------------------------------------------
# BUILD IMAGE SEQUENCE
# --------------------------------------------------

sequence = build_sequence(

    ra=RA,

    dec=DEC,

    fov=FOV,

    detector=detector,

    target_epochs=6,

    min_separation_days=14,

    catalogue=catalogue
)

print("\nREGISTERED OBSERVATIONS")

print(

    sequence[[

        "epoch_id",

        "date",

        "detector",

        "overlap",

        "em_min_um",

        "em_max_um"

    ]].to_string(index=False)
)


# --------------------------------------------------
# DISPLAY REGISTERED IMAGES
# --------------------------------------------------

images = load_image_stack(

    sequence
)

vmin, vmax = get_display_limits(

    images
)

n = len(images)

fig, axes = plt.subplots(

    1,

    n,

    figsize=(4 * n, 5),

    squeeze=False
)

for i, image in enumerate(images):

    ax = axes[0, i]

    ax.imshow(

        image,

        origin="lower",

        cmap="gray",

        vmin=vmin,

        vmax=vmax
    )

    ax.set_title(

        sequence.iloc[i]["date"][:10]
    )

    ax.set_xlabel("X pixel")

    ax.set_ylabel("Y pixel")

fig.suptitle(

    f"Barnard's Star | "
    f"SPHEREx detector {detector}"
)

plt.tight_layout()

plt.show()
