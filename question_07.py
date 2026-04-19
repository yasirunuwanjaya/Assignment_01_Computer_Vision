import cv2
import numpy as np

def zoom_image(image, s, method='nearest'):
    # Calculate new dimensions
    height, width = image.shape[:2]
    new_height, new_width = int(height * s), int(width * s)
    
    if method == 'nearest':
        # (a) Nearest-neighbor
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    elif method == 'bilinear':
        # (b) Bilinear interpolation
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

def compute_ssd(img1, img2):
    # Ensure images are the same size for comparison
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Normalized SSD formula: sum((I1 - I2)^2) / (Total Pixels)
    diff = img1.astype(np.float32) - img2.astype(np.float32)
    ssd = np.sum(np.square(diff)) / (img1.shape[0] * img1.shape[1])
    return ssd

# --- TESTING ---
img_small = cv2.imread('small_version.png', cv2.IMREAD_GRAYSCALE)
img_original = cv2.imread('original_large.png', cv2.IMREAD_GRAYSCALE)

if img_small is None or img_original is None:
    print("Error: Images not found. Check filenames!")
else:
    # Scale up factor (e.g., if original is 4x larger, s = 4.0)
    scale_factor = img_original.shape[0] / img_small.shape[0]

    # Test Nearest Neighbor
    zoomed_nn = zoom_image(img_small, scale_factor, method='nearest')
    ssd_nn = compute_ssd(img_original, zoomed_nn)

    # Test Bilinear
    zoomed_bl = zoom_image(img_small, scale_factor, method='bilinear')
    ssd_bl = compute_ssd(img_original, zoomed_bl)

    print(f"SSD (Nearest-Neighbor): {ssd_nn:.2f}")
    print(f"SSD (Bilinear): {ssd_bl:.2f}")

    # Save results
    cv2.imwrite('q7_zoomed_nn.png', zoomed_nn)
    cv2.imwrite('q7_zoomed_bilinear.png', zoomed_bl)