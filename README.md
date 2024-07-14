# Pixelated-image-detection-and-correction
Pixelated images occur when an image is compressed or resized too much, resulting in a loss of detail and a blocky appearance.

Pixelated images are characterized by the visible square blocks that make up the image. This occurs when the number of pixels is insufficient to represent the original image detail.

Loss of Detail:

The visible blockiness obscures fine lines, textures, and subtle color variations.

Blurred Appearance:

The edges of objects become fuzzy and unclear, diminishing the overall visual quality.

Compression Artifacts:

Pixelation is often a byproduct of image compression algorithms, especially when high compression ratios are used.


The above code checks if the mean difference  exceeds the threshold, if it is more than threshold then it is a pixelated image and it gets corrected.If less than threshold then it is detected as a non pixelated image.

NOTE:
There isn't a universally fixed threshold value for detecting pixelation in images because the optimal threshold can vary based on factors like image resolution, content, and quality. However, you can determine a specific threshold value for your use case through empirical testing and analysis.




Bilateral filter is applied to the input image. This filter smooths the image while preserving edges, using parameters like:d,sigmaColor,
sigmaSpace


![butterfly](https://github.com/user-attachments/assets/ee587e2c-2835-4a6a-95b0-cb44a2b4b692)


Here are the drive links for datasets used in the above code:

original images

https://drive.google.com/drive/folders/1_mdvmEBjBulqE323azVAyWOrI7MF_BZ9?usp=sharing

pixelated images

https://drive.google.com/drive/folders/1Ggavx935Ck4h7Xjy2oYva7ZipZ95Be0q?usp=sharing
