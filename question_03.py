import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the grayscale image
img = cv2.imread('runway.png', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: runway.png not found!")
else:
    # --- MANUAL HISTOGRAM EQUALIZATION ---
    
    # 1. Get image dimensions
    rows, cols = img.shape
    total_pixels = rows * cols

    # 2. Calculate the Histogram (counts of each intensity 0-255)
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])

    # 3. Calculate the Cumulative Distribution Function (CDF)
    # The 'cumsum' function adds up the histogram values step-by-step
    cdf = hist.cumsum()
    
    # 4. Normalize the CDF
    # Formula: (cdf - min_cdf) / (total_pixels - min_cdf) * 255
    cdf_m = np.ma.masked_equal(cdf, 0) # Mask zeros to avoid division errors
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf_final = np.ma.filled(cdf_m, 0).astype('uint8')

    # 5. Map the original image pixels to the new values
    img_equalized = cdf_final[img]

    # --- SAVE RESULTS ---
    cv2.imwrite('q3_equalized.png', img_equalized)

    # --- PLOT COMPARISON ---
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(img.flatten(), 256, [0, 256], color='gray')
    plt.title("Original Histogram")

    plt.subplot(1, 2, 2)
    plt.hist(img_equalized.flatten(), 256, [0, 256], color='blue')
    plt.title("Equalized Histogram")
    
    plt.savefig('q3_histogram_comparison.png')
    print("Question 3 Finished! Saved 'q3_equalized.png'.")
    plt.show()