import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

print("--- Question 2 Debug Started ---")

# 1. Check if the file exists physically
filename = 'fig2.jpg'
if not os.path.exists(filename):
    print(f"ERROR: Cannot find '{filename}' in the folder.")
    print(f"Current folder content: {os.listdir('.')}")
else:
    print(f"Found {filename}. Loading...")
    img = cv2.imread(filename)

    if img is None:
        print("ERROR: OpenCV could not read the image. It might be corrupted.")
    else:
        # 2. LAB Processing
        print("Step 2: Converting to LAB...")
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(lab)

        # 3. Gamma
        print("Step 3: Applying Gamma 0.5...")
        l_norm = l.astype(np.float32) / 255.0
        l_gamma = np.power(l_norm, 0.5)
        l_corrected = (l_gamma * 255).astype(np.uint8)

        # 4. Save Image
        print("Step 4: Merging and Saving...")
        lab_corrected = cv2.merge([l_corrected, a, b])
        img_corrected = cv2.cvtColor(lab_corrected, cv2.COLOR_Lab2BGR)
        cv2.imwrite('q2_result.png', img_corrected)

        # 5. Force Histogram Save (No pop-up window)
        print("Step 5: Generating Histogram file...")
        plt.figure()
        plt.hist(l.flatten(), bins=256, range=[0, 256], color='gray', alpha=0.5, label='Original')
        plt.hist(l_corrected.flatten(), bins=256, range=[0, 256], color='blue', alpha=0.5, label='Corrected')
        plt.legend()
        plt.savefig('q2_histogram_output.png')
        print("DONE! Check your folder for 'q2_result.png' and 'q2_histogram_output.png'")

print("--- Script Finished ---")