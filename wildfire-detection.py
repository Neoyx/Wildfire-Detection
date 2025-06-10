import rasterio
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, floor
import random
from dataclasses import dataclass
import images
import cv2

# Lade die Bänder (alle 20m → gleiche Form)
def load_band(path):
    with rasterio.open(path) as src:
        band = src.read(1).astype(float)
    return band

b12_path, b11_path, b8a_path, b04_path, b03_path, b02_path, cm_path = images.get_band_paths(images.Flin_Flon)

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
outer_fire_mask = (b12_norm > 0.6) & (b11_norm < 0.5) & (b8a_norm < 0.5) # & (cloud_mask == 0) # low b11 and b8a (G, B) to exclude clouds and vegetation
outer_fire_mask = outer_fire_mask.astype(np.uint16)

# dilate the fire to enlarge the detection radius for the core fire
fire_closing = np.ones((135, 135), np.uint16)
#dilated_fire = cv2.dilate(outer_fire_mask.astype(np.uint8), filter, iterations=5)
closed_fire_mask = cv2.morphologyEx(outer_fire_mask, cv2.MORPH_CLOSE, fire_closing)

# search for yellow/white fire-pixels near the red pixels
# & (b04_norm < 0.8) & (b03_norm < 0.8) & (b02_norm < 0.8) ==> Alternative zur cloud_mask == 0
core_fire_mask = (b12_norm > 0.7) & (b11_norm > 0.7) & (closed_fire_mask == 1) & (b04_norm < 0.8) & (b03_norm < 0.8) & (b02_norm < 0.8) # & (cloud_mask == 0)
core_fire_mask = core_fire_mask.astype(np.uint16)

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

kernel = np.ones((7,7),np.uint16)
combinedRegion_closed = cv2.morphologyEx(final_fire_mask, cv2.MORPH_CLOSE, kernel)

kernel2 = np.ones((2,2),np.uint16)
combinedRegion_opened = cv2.morphologyEx(combinedRegion_closed, cv2.MORPH_OPEN, kernel2)

def seq_reg(img, n8):
    img = img.copy()
    (height, width) = img.shape
    out_img = np.zeros((height, width, 3), dtype=np.uint16)
    m = 2
    collisions = set()

    # Pass 1 – Assign Initial Labels
    for v in range(height):
        for u in range(width):
            if img[v, u] == 1:
                neighbors = []

                if u > 0 and img[v, u - 1] > 1: # Nachbar links
                    neighbors.append(img[v, u - 1])

                if v > 0 and img[v - 1, u] > 1: # Nachbar oben
                    neighbors.append(img[v - 1, u])
                
                if n8:
                    if u > 0 and v > 0 and img[v - 1, u - 1] > 1: # Nachbar oben links
                        neighbors.append(img[v - 1, u - 1])
                    if u < width - 1 and v > 0 and img[v - 1, u + 1] > 1: # Nachbar oben rechts
                        neighbors.append(img[v - 1, u + 1])    
                
                unique_neigbors = list(set(neighbors))
                if len(unique_neigbors) == 0: # Alle Nachbarn sind Hintergrundpixel
                    img[v, u] = m
                    m += 1
                elif len(unique_neigbors) == 1: # Genau ein Nachbar hat Labelwert größer 1
                    img[v, u] = unique_neigbors[0]
                else: # Mehrere Nachbarnn haben Labelwert größer 1
                    # Nimm einfach den ersten als neuen Label
                    img[v, u] = unique_neigbors[0]
                    k = unique_neigbors[0]
                    for n1 in unique_neigbors:
                        if n1 != k:
                            collisions.add((n1, k))

    # Pass 2 – Resolve Label Collisions  
    L = range(2, m)
    R = [{i} for i in L]
    for (a, b) in collisions:
        for s in R:
            if a in s:
                r_a = s # der set, der gerade a enthält
            if b in s:
                r_b = s # der set, der gerade b enthält
        if r_a != r_b:
            r_a.update(r_b)
            R.remove(r_b) # WARN: vielleicht nur leer machen

    random.seed(20)
    label_to_color = {}
    for s in R:
        base_label = min(s)
        label_to_color[base_label] = [random.randint(0, 255) for _ in range(3)]
    # Pass 3 - Relabel the Image    
    for v in range(height):
        for u in range(width):
            if img[v, u] > 1:
                for i, s in enumerate(R):
                    if img[v, u] in s:
                        base_label = min(s)
                        out_img[v, u] = label_to_color[base_label]
                        #img[v, u] = min(s)
                        break
    print(len(R))
    return out_img

n8 = True
labeled_fire = seq_reg(combinedRegion_opened, n8)

fullscreen = False
sync_zoom = True

@dataclass
class Subplot:
    title: str
    img: np.ndarray
    cmap: str | None = None

subplots = [
    Subplot("Farbbild (makiert) (B04, B03, B02 – 20m)", color_marked),
    Subplot("Infrarotbild (B12, B11, B8A – 20m)", infrared),
    Subplot("Aktive Feuer-Pixel (weiß)", final_fire_mask, cmap='gray'),
    Subplot("Kombiniertes Feuer (Closed))", combinedRegion_closed, cmap='gray'),
    Subplot("Kombiniertes Feuer (Closed-Open)", combinedRegion_opened, cmap='gray'),
    Subplot("Regionenmarkiertes Feuer", labeled_fire, cmap='gray')
]

# Visualisierung
rows = int(np.ceil(len(subplots) / 3))

plt.subplots(rows, 3, figsize=(16, 8), sharex=sync_zoom, sharey=sync_zoom)

if fullscreen:
    plt.get_current_fig_manager().full_screen_toggle()

for i, subplot in enumerate(subplots, start=1):
    plt.subplot(rows, 3, i)
    plt.imshow(subplot.img, cmap=subplot.cmap)
    plt.title(subplot.title)
    plt.axis('off')

# Hide any remaining empty subplots
for i in range(len(subplots), rows * 3):
    plt.subplot(rows, 3, i + 1)
    plt.axis('off')

plt.show()
