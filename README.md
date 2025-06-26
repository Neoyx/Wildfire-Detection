# Wildfire Detection

This product is meant to detect wildfires on a given satellite image.
We used images of the satellite **Sentinel-2 L2A**, which already filters out particles from the atmosphere and provides a "bottom-of-atmosphere" view. This improves the detection of the fire.
To use our product simply run the file "wildfire-detection.py". 

## Detection
This repository has two different detections.
1. Fire Detection
2. Burnt Area Detecion

### Fire detection
In our implementation we broke down the fire detection into two masks.
 The first mask is the so-called **outer fire mask** which detects the red pixels of the fire.
 The other one is the **core fire mask** which detects the brighter, hotter yellow pixels of the fire and only looks for these pixels in the same area of the red pixels of the fire to reduce false detections.
 Additionally we also filter out clouds using high values in the visible light bands B02, B03 and B04.
The detection mainly works by using different thresholds for the infrared bands. Feel free to mess around with these tresholds :)

Additionally we have segmented the fire using sequential regioning (implementation seen in the respective files). 
You can use this to count the number of fires in the image (useful for debugging the filter and just interesting in general).

### Burnt area detection

#### Why pure thresholding detection does not work
Like in our fire detection we first tried using different tresholds to detect the burnt areas of the image. Normally band B08a would be really good for this since it provides high values for a healthy vegetation and low values if the opposite is true. However since all used images in this repository also include a fire, the smoke of these fires messes with the values of B08a since it cannot penetrate the smoke to provide useful information. Besides bands B12 and B11 have a hard time differentiating the burnt vegetation from healthy one, at least in the images we worked with.

#### Edge detection
While analysing the infrared (B12, B11, B08a) images, we noticed a clear destinction of the burnt and healthy areas. To identify the burnt region more precisely, we decided to apply edge detection techniques.
After contrast boosting the image, we used Sobel and Canny algorithms (OpenCV) to detect the edges, and we also created a combined approach, to improve the detection accuracy. 
Just like in the fire detection both Sobel and Canny use different thresholds in their function that you can fine-tune if you want to adjust the results.

## Own use
If you want to use your own images feel free to download from [Copernicus](https://browser.dataspace.copernicus.eu), which is also where we downloaded the current data.
It's important to only download images from **Sentinel-2 L2A**. The lower the cloud index is, the better is the result of the detection.
Once the image is downloaded, you can find the important bands used for the detection in the "GRANULE" folder after opening the zip file. Go to "GRANULE/{image-name}/IMG_DATA/R20M".

In this folder, you’ll find the bands:
- b04, b03 and b02 used for the true color image
- b12, b11 and b8a used for the infrared image.

Every band in this folder has a resolution of 20m, which means each pixel has a size of 20m x 20m.
Now that you are aware of the bands, you can adjust the path to load them.
