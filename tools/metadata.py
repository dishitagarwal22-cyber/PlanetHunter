
from astropy.io import fits
from astropy.wcs import WCS

with fits.open("data/observation1.fits") as hdul:

    header = hdul["IMAGE"].header

    print("OBSERVATION METADATA")
    print("--------------------")

    keywords = [
        "DATE-OBS",
        "DATE-END",
        "MJD-OBS",
        "MJD-END",
        "EXPTIME",
        "BUNIT",
        "CRVAL1",
        "CRVAL2",
        "CTYPE1",
        "CTYPE2",
        "CTYPE3",
        "CRVAL3",
        "CDELT3",
        "DETECTOR",
        "FILTER"
    ]

    for key in keywords:
        print(f"{key}: {header.get(key, 'Not available')}")

    print("\nWAVELENGTH TABLE")
    print(hdul["WCS-WAVE"].columns)

    print("\nWAVELENGTH-RELATED HEADER ENTRIES")

    for key in header:
        if any(word in key for word in [
            "WAVE", "LAMB", "BAND", "SPEC", "MJD", "DATE"
        ]):
            print(f"{key}: {header[key]}")

    print("\nWCS INFORMATION")

    wcs = WCS(header, fobj=hdul)
    print(wcs)
