import numpy as np
from sequential_regioning import sequential_regioning
import visualisation
from visualisation import Subplot, SliderConfig
import time
import images
from bands import get_normalized_bands
import cv2
from scipy.ndimage import gaussian_filter, laplace
from skimage.feature import canny
from skimage import measure, morphology
from skimage.exposure import rescale_intensity
import sequential_regioning_cpp

# Stacking the image
def stack_img(band_1, band_2, band_3):
    img = np.dstack((band_1, band_2, band_3)) # Stacking the bands
    img /= np.percentile(img, 98) # Optimize contrast
    img = np.clip(img, 0, 1) # Normalize
    return img

def update_img(orignal_band, value, base_img, col = [0, 1, 0], dilate_size=50):
    mask = orignal_band > value
    if dilate_size > 0:
        area_dilatation = np.ones((dilate_size, dilate_size), np.uint16)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_DILATE, area_dilatation)
    band = base_img.copy()
    band[mask > 0] = col
    return band

def main(img: images.Image, plot_sync_zoom: bool = True, down_scale: bool = True, down_scale_factor: int = 2):
    time_start = time.time()

    # Loading the bands
    b12_norm, b11_norm, b8a_norm, b04_norm, b03_norm, b02_norm, cm_norm = get_normalized_bands(img)

    
    # Downscaling the images if required
    if down_scale:
        f = down_scale_factor
        b12_norm = cv2.resize(b12_norm, (b12_norm.shape[1] // f, b12_norm.shape[0] // f), interpolation=cv2.INTER_LINEAR)
        b11_norm = cv2.resize(b11_norm, (b11_norm.shape[1] // f, b11_norm.shape[0] // f), interpolation=cv2.INTER_LINEAR)
        b8a_norm = cv2.resize(b8a_norm, (b8a_norm.shape[1] // f, b8a_norm.shape[0] // f), interpolation=cv2.INTER_LINEAR)
        b04_norm = cv2.resize(b04_norm, (b04_norm.shape[1] // f, b04_norm.shape[0] // f), interpolation=cv2.INTER_LINEAR)
        b03_norm = cv2.resize(b03_norm, (b03_norm.shape[1] // f, b03_norm.shape[0] // f), interpolation=cv2.INTER_LINEAR)
        b02_norm = cv2.resize(b02_norm, (b02_norm.shape[1] // f, b02_norm.shape[0] // f), interpolation=cv2.INTER_LINEAR)

    # Stacking the bands
    infrared = stack_img(b12_norm, b11_norm, b8a_norm)
    color = stack_img(b04_norm, b03_norm, b02_norm)
    

    # 1. Fire-detection

    # Low b11 and b8a (G, B) to exclude clouds and vegetation
    #outer_fire_mask = (b12_norm > 0.5) & (b11_norm < 0.4) & (b8a_norm < 0.5)
    outer_fire_mask = (b12_norm > 0.3) & (b11_norm < 0.4) & (b8a_norm < 0.4)
    outer_fire_mask = outer_fire_mask.astype(np.uint16)

    # TODO: Suppression mode ==> Run filter above outer fire mask and check if it is a single pixel, then discard

    # Dilate the fire to enlarge the detection radius for the core fire
    fire_closing = np.ones((135, 135), np.uint16)
    closed_fire_mask = cv2.morphologyEx(outer_fire_mask, cv2.MORPH_CLOSE, fire_closing)

    # Search for yellow/white fire pixels near the red pixels
    core_fire = (b12_norm > 0.5) & (b11_norm > 0.2) & (closed_fire_mask == 1) # searching for yellow fire pixels
    cloud_filter = (b04_norm < 0.6) & (b03_norm < 0.6) & (b02_norm < 0.6) # filtering the clouds
    core_fire_mask = core_fire & cloud_filter
    core_fire_mask = core_fire_mask.astype(np.uint16)

    # TODO: Problem ==> Fehldetektierte rot-angestrahlte Wolken herausfiltern
    # Problem an Anwendung von Cloudmask: Unter den Wolken liegende Feuer werden nicht mehr erkannt

    # TODO: Groesse der Filtermasken evtl. je nach Groesse des Feuers dynamisch anpassen

    # Storing the indices of the fire masks
    outer_fire_indices = np.where(outer_fire_mask)
    core_fire_indices = np.where(core_fire_mask)

    # Mark the fire in the color image in red and yellow
    color_marked = color.copy()
    color_marked[outer_fire_indices[0], outer_fire_indices[1]] = [1, 0, 0] # Mark the outer fire in red
    color_marked[core_fire_indices[0], core_fire_indices[1]] = [1, 1, 0] # Mark the core fire in yellow

    def update_ultimate_threshoold_finder(b12_value, b11_value, b8a_value, base_image):
        image = base_image.copy()

        red = np.array([1.0, 0.0, 0.0])
        green = np.array([0.0, 1.0, 0.0])
        blue = np.array([0.0, 0.0, 1.0])

        mask_b12 = b12_norm > b12_value
        mask_b11 = b11_norm > b11_value
        mask_b8a = b8a_norm > b8a_value

        image[mask_b12] = [0, 0, 0]
        image[mask_b11] = [0, 0, 0]
        image[mask_b8a] = [0, 0, 0]

        image[mask_b12] = image[mask_b12] + red
        image[mask_b11] = image[mask_b11] + green
        image[mask_b8a] = image[mask_b8a] + blue
        return image

    # Combine the fire masks
    final_fire_mask = outer_fire_mask | core_fire_mask

    # Increasing the size of the fire marks for visualization
    visual_dilatation = np.ones((50, 50), np.uint16)
    final_fire_mask_dilated = cv2.morphologyEx(final_fire_mask.astype(np.uint8), cv2.MORPH_DILATE, visual_dilatation)
    final_fire_indices = np.where(final_fire_mask_dilated)   
    color_marked_dilated = color.copy()
    color_marked_dilated[final_fire_indices[0], final_fire_indices[1]] = [1, 0, 0]
    

    # 2. Regioning of the fires

    kernel_medium = np.ones((7,7),np.uint16)
    combined_region_closed = cv2.morphologyEx(final_fire_mask, cv2.MORPH_CLOSE, kernel_medium)

    kernel_small = np.ones((2,2),np.uint16)
    combined_region_opened = cv2.morphologyEx(combined_region_closed, cv2.MORPH_OPEN, kernel_small)

    # labeled_fire, amount_regions = sequential_regioning(combinedRegion_opened, n8=True) # using the pure Python implementation
    labeled_fire, amount_regions  = sequential_regioning_cpp.run(combined_region_opened, n8=True) # using the C++ implementation for performance


    # 3. Detecting the burned area

    # Converting the bands to greyscale by dividing b12 with b11 for brightening the burned area
    burn_index = b12_norm / (b11_norm + 1e-6) # added constant denominator to avoid dividing by zero

    # Normalize burn_index to 0–1
    burn_index = (burn_index - burn_index.min()) / (burn_index.max() - burn_index.min() + 1e-6)

    # Filtering sharp edges that aren't part of the burned area
    mean_mask = ((burn_index > 0) & (b11_norm >= 0.2)) # avoid black areas or water for computing the mean
    mean = burn_index[mean_mask].mean()
    burn_index[b11_norm < 0.2] = mean     # filtering water
    burn_index[b8a_norm > 0.5] = mean     # filtering vegetation
    burn_index[b11_norm > 0.3] = mean     # filtering bright areas

    # Gamma correction to make the image brighter
    burn_index = burn_index**0.5 

    # Applying sobel operator for edge detection of the burned area
    smoothed = gaussian_filter(burn_index, sigma=2.0) # applying gaussian filter to smooth small edges
    sobelx = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3) # in x-direction
    sobely = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3) # in y-direction
    edges_sobel = np.hypot(sobelx, sobely) # combine filters
    edges_sobel = cv2.normalize(edges_sobel, None, 0, 1, cv2.NORM_MINMAX) # normalize
    binary_edges_sobel = edges_sobel > np.percentile(edges_sobel, 98) # only keep strongest edges

    # Applying canny operator for detecting sharp edges only
    edges_canny = canny(burn_index, sigma=2.0, low_threshold=np.percentile(edges_sobel, 90), high_threshold=np.percentile(edges_sobel, 95)) 
    dilated_edges = cv2.morphologyEx(edges_canny.astype(np.uint8), cv2.MORPH_DILATE, kernel_medium)

    # Dilate detected edges by canny in order to limit the detection radius of sobel
    area_dilatation = np.ones((200, 200), np.uint16)
    dilated_edges_canny = cv2.morphologyEx(dilated_edges.astype(np.uint8), cv2.MORPH_DILATE, area_dilatation)

    combined_edges = binary_edges_sobel & dilated_edges_canny # Keep all sobel pixels that overlap dilated canny area
    combined_edges_closed = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel_medium) # Close holes in burned area
    combined_edges_opened = cv2.morphologyEx(combined_edges_closed, cv2.MORPH_OPEN, kernel_small) # Remove artifacts

    # Computing the size of the burning area
    number_fire_pixels = np.count_nonzero(final_fire_mask)
    burning_area = 400 * number_fire_pixels / 1000000
    print(f"Size of active fire area: {burning_area}km^2")

    time_end = time.time()
    print(f"Complete Processing time: {time_end - time_start:.2f} seconds")


    subplots_data = [
        # Subplot("Farbbild (B04, B03, B02 – 20m)", color),
        # Subplot("Farbbild (makiert) (B04, B03, B02 – 20m)", color_marked),
        # Subplot("Farbbild (makiert - groß) (B04, B03, B02 – 20m)", color_marked_dilated),
        # Subplot("Infrarotbild (B12, B11, B8A – 20m)", infrared),
        #Subplot("Aktive Feuer-Pixel (weiß)", final_fire_mask, cmap='gray'),
        #Subplot("Kombiniertes Feuer (Closed))", combinedRegion_closed, cmap='gray'),
        #Subplot("Kombiniertes Feuer (Closed-Open)", combinedRegion_opened, cmap='gray'),
        #Subplot(f"Regionenmarkiertes Feuer | Regions: {amount_regions}", labeled_fire, cmap='gray'),
        #Subplot("Verbrannte Fläche", burn_index, cmap='gray'),
        # Subplot("Verbrannte Fläche (Sobel)", binary_edges_sobel, cmap='gray'),
        # Subplot("Verbrannte Fläche (Canny)", dilated_edges, cmap='gray'),
        # Subplot("Verbrannte Fläche (kombiniert)", combined_edges_opened, cmap='gray'),
        # Subplot("b12_norm", b12_norm, cmap='hot'),
        # Subplot("b11_norm", b11_norm, cmap='hot'),
        # Subplot("b8a_norm", b8a_norm, cmap='hot'),
        Subplot(
            "b12_norm (markiert)",
            infrared,
            slider_configs=[SliderConfig(initial_value=0.5)],
            slider_update_function=lambda x: update_img(b12_norm, x[0], base_img=infrared, col=[0, 1, 0], dilate_size=0)
        ),
        Subplot(
            "b11_norm (markiert)",
            infrared,
            slider_configs=[SliderConfig(initial_value=0.5)],
            slider_update_function=lambda x: update_img(b11_norm, x[0], base_img=infrared, col=[0, 1, 0])
        ),
        Subplot(
            "b8a_norm (markiert)",
            infrared,
            slider_configs=[SliderConfig(initial_value=0.5)],
            slider_update_function=lambda x: update_img(b8a_norm, x[0], base_img=infrared, col=[0, 1, 0])
        ),
        Subplot(
            "ultimate threshold finder",
            infrared,
            slider_configs= [
                SliderConfig(initial_value=0.5, label="B12 Red"),
                SliderConfig(initial_value=0.5, label="B11 Green"),
                SliderConfig(initial_value=0.5, label="B8A Blue")
            ],
            slider_update_function=lambda x: update_ultimate_threshoold_finder(*x, base_image=infrared)
        ),
    ]

    # subplots_data_2 = [
    #     Subplot("color", color, cmap='hot'),
    #     Subplot("infrared", infrared, cmap='hot'),
    #     Subplot("burn index", burn_index, cmap='hot'),
    #     Subplot("b12_norm", b12_norm, cmap='hot'),
    #     Subplot("b11_norm", b11_norm, cmap='hot'),
    #     Subplot("b8a_norm", b8a_norm, cmap='hot'),
    # ]
    
    visualisation.plot(subplots_data, plot_sync_zoom=plot_sync_zoom)


if __name__ == "__main__":
    main(
        images.Park_Fire_2,  # Change to any image from the images module
        plot_sync_zoom=True,  # Set to False to disable synchronized zooming
        down_scale=True,  # Set to False to disable downscaling of the images
        down_scale_factor=4  # Factor by which the images are downscaled (2 means half the size)
    )