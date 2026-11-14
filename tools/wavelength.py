
from astropy.io import fits
import numpy as np

with fits.open("data/observation1.fits") as hdul:

    table = hdul["WCS-WAVE"].data

    x = table["X"][0]
    y = table["Y"][0]
    values = table["VALUES"][0]

    print("X control points:")
    print(x)

    print("\nY control points:")
    print(y)

    print("\nValues shape:")
    print(values.shape)

    print("\nCentral wavelength grid:")
    print(values[:, :, 0])

    print("\nBandwidth grid:")
    print(values[:, :, 1])

    print("\nSpectral table header:")
    print(repr(hdul["WCS-WAVE"].header))
