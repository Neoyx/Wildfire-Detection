###Wildfire Detection

This project is meant to detect wildfires on a given satellite image.
We used images of the satellite Sentinel L2A, which already filters out particles from the atmosphere and provides a "bottom-of-atmosphere" view. This improves the detection of the fire.
To use our product, you first have to download an image from Copernicus (https://browser.dataspace.copernicus.eu) on which you want to detect wildfires.
It's important to only download images from Sentinel L2A. The lower the cloud index is, the better is the result of the detection.
Once the image is downloaded, you can find the important bands used for the detection in the "GRANULE" folder after opening the zip file. Go to "GRANULE/<image-name>/IMG_DATA/R20M".
There you can find the bands b04, b03 and b02 used for the color image and b12, b11 and b8a used for the infrared image. For the detection of the fire, we mostly used the infrared bands. 
Every band in this folder has a resolution of 20m, which means each pixel has a size of 20x20m.
Now that you are aware of the bands, you can adjust the path to load them. Then simply run the product and start detecting.
