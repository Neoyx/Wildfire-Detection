import rasterio
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, floor

# Pfade zu den jp2-Dateien (Infrarot)
b12_path = "images/2025_Flin_Flon/infrared/T13UFA_20250602T175931_B12_20m.jp2"
b11_path = "images/2025_Flin_Flon/infrared/T13UFA_20250602T175931_B11_20m.jp2"
b8a_path = "images/2025_Flin_Flon/infrared/T13UFA_20250602T175931_B8a_20m.jp2"

# Pfade zu den jp2-Dateien (True-Color)
b04_path = "images/2025_Flin_Flon/color/T13UFA_20250602T175931_B04_20m.jp2"
b03_path = "images/2025_Flin_Flon/color/T13UFA_20250602T175931_B03_20m.jp2"
b02_path = "images/2025_Flin_Flon/color/T13UFA_20250602T175931_B02_20m.jp2"

cm_path = "images/2025_Flin_Flon/MSK_CLDPRB_20m.jp2"

# Lade die Bänder (alle 20m → gleiche Form)
def load_band(path):
    with rasterio.open(path) as src:
        band = src.read(1).astype(float)
    return band

b12 = load_band(b12_path)
b11 = load_band(b11_path)
b8a = load_band(b8a_path)
b04 = load_band(b04_path)
b03 = load_band(b03_path)
b02 = load_band(b02_path)

cloud_mask = load_band(cm_path)

# Normalisiere Bänder auf Wertebereich zwischen 0 und 1
def normalize_band(band):
    band = band.astype(float)
    band /= 10000.0  # Sentinel-2 typische Skalierung
    band = np.clip(band, 0, 1)
    return band

b12_norm = normalize_band(b12)
b11_norm = normalize_band(b11)
b8a_norm = normalize_band(b8a)
b04_norm = normalize_band(b04)
b03_norm = normalize_band(b03)
b02_norm = normalize_band(b02)

cloud_mask = normalize_band(cloud_mask)

# Stacking und Normalisierung des Farbbildes
def stack_img(band_1, band_2, band_3):
    img = np.dstack((band_1, band_2, band_3)) # Stacking, um aus 3 Graustufenbildern ein Farbbild zu machen
    img /= np.percentile(img, 98) # Kontrast optimieren durch Normalisierung des Bildes
    img = np.clip(img, 0, 1)
    return img

infrared = stack_img(b12_norm, b11_norm, b8a_norm)
color = stack_img(b04_norm, b03_norm, b02_norm)

# Detektion des Feuers
threshold = 0.5
fire_mask = b12_norm > threshold # Schaue nur nach B12-Band, da Feuer in diesem Wellenlängenbereich am stärksten reflektiert
fire_mask = fire_mask & (cloud_mask == 0)
fire_indices = np.where(fire_mask)

# Markierung des Feuers im Farbbild in rot
color_marked = color.copy()
color_marked[fire_indices[0], fire_indices[1]] = [1, 0, 0]

# TODO: rot markierten Feuer-Pixel im Farbbild evtl. durch Dilatation oder andere Filter vergroessern

# Visualisierung
plt.figure(figsize=(16, 8))
plt.subplot(1, 3, 1)
plt.imshow(color_marked)
plt.title("Farbbild (B04, B03, B02 – 20m)")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(infrared)
plt.title("Infrarotbild (B12, B11, B8A – 20m)")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(fire_mask, cmap='gray')
plt.title("Aktive Feuer-Pixel (weiß)")
plt.axis('off')
plt.show()
