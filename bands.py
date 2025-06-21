import rasterio
import numpy as np
import images
import os

def load_band(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        band = src.read(1).astype(float)
    return band

# Normalize bands between 0 and 1
def normalize_band(band: np.ndarray) -> np.ndarray:
    band = band.astype(float)
    # band /= 10000.0  
    band_min = np.min(band)
    band_max = np.max(band)
    if band_max - band_min > 0:
        band = (band - band_min) / (band_max - band_min)
    return band

def get_bands(img: images.Image):
    cm = None
    b12_path, b11_path, b8a_path, b04_path, b03_path, b02_path, cm_path = images.get_band_paths(img)

    scaling = 10000.0 # Sentinel-2 typical scaling

    b12 = load_band(b12_path) / scaling
    b11 = load_band(b11_path) / scaling
    b8a = load_band(b8a_path) / scaling
    b04 = load_band(b04_path) / scaling
    b03 = load_band(b03_path) / scaling
    b02 = load_band(b02_path) / scaling
    if os.path.exists(cm_path): cm = load_band(cm_path) /scaling
    
    return b12, b11, b8a, b04, b03, b02, cm