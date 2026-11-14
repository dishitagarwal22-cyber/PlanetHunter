
import requests
from pathlib import Path
from astropy.io import fits

url = (
    "https://irsa.ipac.caltech.edu/ibe/data/spherex/"
    "qr3/level2/2026W21_2A/l2b-v27-2026-268/1/"
    "level2_2026W21_2A_0001_2D1_spx_l2b-v27-2026-268.fits"
)

params = {
    "center": "337.8783221,-5.1891588",
    "size": "0.1"
}

response = requests.get(url, params=params, timeout=120)
response.raise_for_status()

path = Path("data/observation1.fits")
path.parent.mkdir(exist_ok=True)
path.write_bytes(response.content)

print("Download complete!")
print("File size:", path.stat().st_size, "bytes")

with fits.open(path) as hdul:
    hdul.info()
