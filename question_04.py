import cv2
import numpy as np

# Use the exact name from your folder. 
filename = 'figer3_women.jpg' 

img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"❌ ERROR: Still cannot read '{filename}'.")
    print("Make sure the file is in C:\\AI\\Assignment_01_Computer_Vision")
else:
    # Applying Otsu's Thresholding
    # ret is the threshold value, th is the binary image
    ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    print(f"✅ Success! Otsu's threshold value is: {ret}")
    
    # Save the result
    cv2.imwrite('q4_otsu_result.png', th)
    print("Output saved as 'q4_otsu_result.png'")