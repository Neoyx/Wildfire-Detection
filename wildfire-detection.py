import rasterio
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, floor
import cv2

img_path = "2025_Flin_Flon"
img_name = "T13UFA_20250602T175931"

# Pfade zu den jp2-Dateien (Infrarot)
b12_path = f"images/{img_path}/infrared/{img_name}_B12_20m.jp2"
b11_path = f"images/{img_path}/infrared/{img_name}_B11_20m.jp2"
b8a_path = f"images/{img_path}/infrared/{img_name}_B8a_20m.jp2"

# Pfade zu den jp2-Dateien (True-Color)
b04_path = f"images/{img_path}/color/{img_name}_B04_20m.jp2"
b03_path = f"images/{img_path}/color/{img_name}_B03_20m.jp2"
b02_path = f"images/{img_path}/color/{img_name}_B02_20m.jp2"

cm_path = f"images/{img_path}/MSK_CLDPRB_20m.jp2"

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
outer_fire_mask = (b12_norm > 0.6) & (b11_norm < 0.5) & (b8a_norm < 0.5) # & (cloud_mask == 0) # low b11 and b8a (G, B) to exclude clouds and vegetation
outer_fire_mask = outer_fire_mask.astype(np.uint8)

# dilate the fire to enlarge the detection radius for the core fire
fire_closing = np.ones((135, 135), np.uint8)
#dilated_fire = cv2.dilate(outer_fire_mask.astype(np.uint8), filter, iterations=5)
closed_fire_mask = cv2.morphologyEx(outer_fire_mask, cv2.MORPH_CLOSE, fire_closing)

# search for yellow/white fire-pixels near the red pixels
# & (b04_norm < 0.8) & (b03_norm < 0.8) & (b02_norm < 0.8) ==> Alternative zur cloud_mask == 0
core_fire_mask = (b12_norm > 0.7) & (b11_norm > 0.7) & (closed_fire_mask == 1) & (b04_norm < 0.8) & (b03_norm < 0.8) & (b02_norm < 0.8) # & (cloud_mask == 0)
core_fire_mask = core_fire_mask.astype(np.uint8)

# TODO: Fehldetektierte rot-angestrahlte Wolken herausfiltern
# Problem an Anwendung von Cloudmask: Unter den Wolken liegende Feuer werden nicht mehr erkannt
# ==> Eventuelle Loesung: Cloud-Closing (siehe fortfolgend)

# apply another closing to fill the holes created by the cloudmask
#cloud_closing = np.ones((5, 5), np.uint8)
#closed_core_fire_mask = cv2.morphologyEx(core_fire_mask, cv2.MORPH_CLOSE, cloud_closing)
# Problem: Unter Wolken liegende Feuer können NICHT reproduziert werden

# TODO: Groesse der Filtermasken evtl. je nach Groesse des Feuers dynamisch anpassen

outer_fire_indices = np.where(outer_fire_mask)
core_fire_indices = np.where(core_fire_mask)

final_fire_mask = outer_fire_mask | core_fire_mask

# Markierung des Feuers im Farbbild in rot
color_marked = color.copy()
color_marked[outer_fire_indices[0], outer_fire_indices[1]] = [1, 0, 0]
color_marked[core_fire_indices[0], core_fire_indices[1]] = [1, 1, 0]

print(outer_fire_indices)
print(core_fire_indices)

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
plt.imshow(final_fire_mask, cmap='gray')
plt.title("Aktive Feuer-Pixel (weiß)")
plt.axis('off')
plt.show()
