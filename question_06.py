import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- (b) Compute Normalized 5x5 Derivative Kernels ---
def get_dog_kernels(size, sigma):
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    x, y = np.meshgrid(ax, ax)
    
    # Standard Gaussian G(x,y)
    g = (1 / (2 * np.pi * sigma**2)) * np.exp(-0.5 * (x**2 + y**2) / sigma**2)
    
    # Partial Derivatives
    gx = -(x / sigma**2) * g
    gy = -(y / sigma**2) * g
    
    return gx, gy

gx_5x5, gy_5x5 = get_dog_kernels(5, 2)
print("--- 5x5 Derivative Kernel (Gx) ---")
print(gx_5x5)

# --- (c) Visualize 51x51 Gx Kernel as 3D Surface ---
gx_51, _ = get_dog_kernels(51, 2)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(range(51), range(51))
ax.plot_surface(X, Y, gx_51, cmap='RdBu_r') # Red-Blue shows positive/negative peaks
ax.set_title("3D Surface of Derivative-of-Gaussian (Gx)")
plt.savefig('q6_3d_dog.png')
plt.show()

# --- (d) Apply Manual Kernels ---
img = cv2.imread('runway.png', cv2.IMREAD_GRAYSCALE)
if img is not None:
    # Use filter2D to apply our custom Gx and Gy
    grad_x_manual = cv2.filter2D(img.astype(float), -1, gx_5x5)
    grad_y_manual = cv2.filter2D(img.astype(float), -1, gy_5x5)

    # --- (e) OpenCV Sobel Comparison ---
    # Sobel is a standard approximation of the derivative
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

    # Save results (using absolute values to see the edges better)
    cv2.imwrite('q6_grad_x_manual.png', cv2.convertScaleAbs(grad_x_manual))
    cv2.imwrite('q6_sobel_x.png', cv2.convertScaleAbs(sobel_x))
    print("Question 6 images saved!")