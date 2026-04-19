import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- (a) Compute 5x5 Gaussian Kernel (sigma = 2) ---
def get_gaussian_kernel(size, sigma):
    # Create a coordinate grid
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    x, y = np.meshgrid(ax, ax)
    
    # Gaussian formula: G(x,y) = (1/2*pi*sigma^2) * exp(-(x^2 + y^2)/(2*sigma^2))
    kernel = np.exp(-0.5 * (np.square(x) + np.square(y)) / np.square(sigma))
    
    # Normalize so the sum of all coefficients is 1
    return kernel / np.sum(kernel)

kernel_5x5 = get_gaussian_kernel(5, 2)
print("--- 5x5 Gaussian Kernel (sigma=2) ---")
print(kernel_5x5)

# --- (b) Visualize 51x51 Kernel as 3D Surface ---
kernel_51x51 = get_gaussian_kernel(51, 2)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
x, y = np.meshgrid(range(51), range(51))
ax.plot_surface(x, y, kernel_51x51, cmap='viridis')
ax.set_title("51x51 Gaussian Kernel Surface (sigma=2)")
plt.savefig('q5_3d_surface.png')
plt.show()

# --- (c) & (d) Apply Smoothing ---
img = cv2.imread('runway.png', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("❌ Error: runway.png not found!")
else:
    # (c) Manual Filtering using cv2.filter2D (uses our custom kernel)
    img_manual = cv2.filter2D(img, -1, kernel_5x5)
    
    # (d) OpenCV Built-in
    img_opencv = cv2.GaussianBlur(img, (5, 5), 2)

    # Save results for comparison
    cv2.imwrite('q5_manual_blur.png', img_manual)
    cv2.imwrite('q5_opencv_blur.png', img_opencv)
    print("✅ Question 5 results saved!")