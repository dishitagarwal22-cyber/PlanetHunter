
from astropy.io import fits
import numpy as np

for filename in [
    "data/observation1.fits",
    "data/observation2.fits"
]:

    with fits.open(filename) as hdul:

        header = hdul["IMAGE"].header

        table = hdul["WCS-WAVE"].data

        values = table["VALUES"][0]

        wavelength = values[:, :, 0]
        bandwidth = values[:, :, 1]

        print("\nFILE:", filename)

        print("Date:", header["DATE-OBS"])
        print("Detector:", header["DETECTOR"])

        print("Wavelength range:",
              np.nanmin(wavelength),
              np.nanmax(wavelength))

        print("Median wavelength:",
              np.nanmedian(wavelength))

        print("Median bandwidth:",
              np.nanmedian(bandwidth))

        print("X grid:", table["X"][0])
        print("Y grid:", table["Y"][0])
