import rasterio
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, floor
from dataclasses import dataclass

img_path = "2025_Malibu_Wildfire"
img_name = "T11SLT_20250107T183649"

# Pfade zu den jp2-Dateien (Infrarot)
b12_path = f"images/{img_path}/infrared/{img_name}_B12_20m.jp2"
b11_path = f"images/{img_path}/infrared/{img_name}_B11_20m.jp2"
b8a_path = f"images/{img_path}/infrared/{img_name}_B8a_20m.jp2"

# Pfade zu den jp2-Dateien (True-Color)
b04_path = f"images/{img_path}/color/{img_name}_B04_20m.jp2"
b03_path = f"images/{img_path}/color/{img_name}_B03_20m.jp2"
b02_path = f"images/{img_path}/color/{img_name}_B02_20m.jp2"

# cm_path = f"images/{img_path}/MSK_CLDPRB_20m.jp2"

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

# cloud_mask = load_band(cm_path)

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

# cloud_mask = normalize_band(cloud_mask)

# Stacking und Normalisierung des Farbbildes
def stack_img(band_1, band_2, band_3):
    img = np.dstack((band_1, band_2, band_3)) # Stacking, um aus 3 Graustufenbildern ein Farbbild zu machen
    img /= np.percentile(img, 98) # Kontrast optimieren durch Normalisierung des Bildes
    img = np.clip(img, 0, 1)
    return img

infrared = stack_img(b12_norm, b11_norm, b8a_norm)
swir_composite = stack_img(b12_norm, b8a_norm, b04_norm) # SWIR Composite
nir_swir_composite = stack_img(b8a_norm, b11_norm, b04_norm) # NIR-SWIR Composite

color = stack_img(b04_norm, b03_norm, b02_norm)

# Detektion des Feuers
fire_mask = (b12_norm > 0.6) & (b11_norm < 0.5) & (b8a_norm < 0.5) # low b11 and b8a (G, B) to exclude clouds and vegetation
#fire_mask = b12_norm > 0.7
#fire_mask = fire_mask & (cloud_mask == 0)


fire_indices = np.where(fire_mask)

# Markierung des Feuers im Farbbild in rot
color_marked = color.copy()
color_marked[fire_indices[0], fire_indices[1]] = [1, 0, 0]


# TODO: rot markierten Feuer-Pixel im Farbbild evtl. durch Dilatation oder andere Filter vergroessern


fullscreen = True

@dataclass
class Subplot:
    title: str
    img: np.ndarray
    cmap: str | None = None

subplots = [
    Subplot("Farbbild (makiert) (B04, B03, B02 – 20m)", color_marked),
    Subplot("Infrarotbild (B12, B11, B8A – 20m)", infrared),
    Subplot("Aktive Feuer-Pixel (weiß)", fire_mask, cmap='gray'),
    Subplot("Farbbild (B04, B03, B02 – 20m)", color),
    Subplot("SWIR Composite (B12, B8A, B04)", swir_composite),
    Subplot("NIR-SWIR Composite (B8A, B11, B04)", nir_swir_composite),
]

# Visualisierung
rows = int(np.ceil(len(subplots) / 3))

if fullscreen:
    plt.get_current_fig_manager().full_screen_toggle()
else:
    plt.figure(figsize=(16, 8))

for i, subplot in enumerate(subplots, start=1):
    plt.subplot(rows, 3, i)
    plt.imshow(subplot.img, cmap=subplot.cmap)
    plt.title(subplot.title)
    plt.axis('off')
plt.show()
