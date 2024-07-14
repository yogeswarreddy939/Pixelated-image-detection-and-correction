# Pixelated-image-detection-and-correction
Pixelated images occur when an image is compressed or resized too much, resulting in a loss of detail and a blocky appearance.


The above code checks if the mean difference  exceeds the threshold, if it is more than threshold then it is a pixelated image and it gets corrected.If less than threshold then it is detected as a non pixelated image


****ilateral filter is applied to the input image. This filter smooths the image while preserving edges, using parameters like:d,sigmaColor,
sigmaSpace
