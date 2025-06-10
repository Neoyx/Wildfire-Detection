import numpy as np
from sequential_regioning import sequential_regioning
import visualisation
from visualisation import Subplot
import time
import images
from bands import get_normalized_bands
import cv2

# Stacking und Normalisierung des Farbbildes
def stack_img(band_1, band_2, band_3):
    img = np.dstack((band_1, band_2, band_3)) # Stacking, um aus 3 Graustufenbildern ein Farbbild zu machen
    img /= np.percentile(img, 98) # Kontrast optimieren durch Normalisierung des Bildes
    img = np.clip(img, 0, 1)
    return img


def main(img: images.Image, plot_sync_zoom: bool = True):
    time_start = time.time()

    b12_norm, b11_norm, b8a_norm, b04_norm, b03_norm, b02_norm, cloud_mask_norm = get_normalized_bands(img)

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

    # TODO: rot markierten Feuer-Pixel im Farbbild evtl. durch Dilatation oder andere Filter vergroessern

    kernel = np.ones((7,7),np.uint16)
    combinedRegion_closed = cv2.morphologyEx(final_fire_mask, cv2.MORPH_CLOSE, kernel)

    kernel2 = np.ones((2,2),np.uint16)
    combinedRegion_opened = cv2.morphologyEx(combinedRegion_closed, cv2.MORPH_OPEN, kernel2)

    labeled_fire = sequential_regioning(combinedRegion_opened, n8=True)

    time_end = time.time()
    print(f"Complete Processing time: {time_end - time_start:.2f} seconds")


    subplots_data = [
        Subplot("Farbbild (makiert) (B04, B03, B02 – 20m)", color_marked),
        Subplot("Infrarotbild (B12, B11, B8A – 20m)", infrared),
        Subplot("Aktive Feuer-Pixel (weiß)", final_fire_mask, cmap='gray'),
        Subplot("Kombiniertes Feuer (Closed))", combinedRegion_closed, cmap='gray'),
        Subplot("Kombiniertes Feuer (Closed-Open)", combinedRegion_opened, cmap='gray'),
        Subplot("Regionenmarkiertes Feuer", labeled_fire, cmap='gray')
    ]
    
    visualisation.plot(subplots_data, plot_sync_zoom=plot_sync_zoom)


if __name__ == "__main__":
    main(
        images.Flin_Flon,  # Change to any image from the images module
        plot_sync_zoom=True  # Set to False to disable synchronized zooming
    )