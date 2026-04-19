import cv2
import numpy as np

print("--- Question 1: Processing Started ---")

# 1. Load the image
img = cv2.imread('runway.png', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("❌ ERROR: 'runway.png' not found!")
else:
    # 2. Normalize to [0, 1]
    r = img.astype(np.float32) / 255.0

    # (a) Gamma Correction gamma=0.5
    s_gamma_05 = np.power(r, 0.5)

    # (b) Gamma Correction gamma=2.0
    s_gamma_2 = np.power(r, 2.0)

    # (c) Contrast Stretching
    r1, r2 = 0.2, 0.8
    s_stretched = np.clip((r - r1) / (r2 - r1), 0, 1)

    # 3. Save all results as images for your report
    cv2.imwrite('q1_a_gamma05.png', (s_gamma_05 * 255).astype(np.uint8))
    cv2.imwrite('q1_b_gamma2.png', (s_gamma_2 * 255).astype(np.uint8))
    cv2.imwrite('q1_c_stretched.png', (s_stretched * 255).astype(np.uint8))

    print("✅ All images saved successfully!")
    print("Files created: q1_a_gamma05.png, q1_b_gamma2.png, q1_c_stretched.png")
    print("--- Question 1: Finished ---")