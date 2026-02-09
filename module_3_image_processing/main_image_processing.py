import numpy as np
import matplotlib.pyplot as plt
import cv2

# create a 2D Gaussian Kernel
def get_gaussian_kernel(size, sigma):

    # Create a grid of (x, y) coordinates where (0,0) is the center
    center = size // 2
    kernel = np.zeros((size, size))
    
    for x in range(size):
        for y in range(size):
            # Calculate distance from center
            diff = (x - center)**2 + (y - center)**2
            # Gaussian formula
            kernel[x, y] = np.exp(-diff / (2 * sigma**2))
            
    # Normalize the kernel so the sum of all elements is 1.
    return kernel / np.sum(kernel)

# Visualize Fourier Spectrum
def get_fourier_spectrum(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f) # Shift zero freq to center
    # Use log scale because the range of values is huge
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    return magnitude_spectrum


# MAIN EXECUTION
image_path = '/Users/bbimali1/Documents/Computer_Vision_Spring_26/module_3_image_processing/images/grayscale-cat.jpg' 
loaded_image = cv2.imread(image_path, 0)
image_size = 512
loaded_image = cv2.resize(loaded_image, (image_size, image_size))
original_image = np.float32(loaded_image)

# Create the Gaussian Kernel
kernel_size = 10
sigma = 3
kernel = get_gaussian_kernel(kernel_size, sigma)

# SPATIAL DOMAIN (CONVOLUTION)
spatial_blur = cv2.filter2D(original_image, -1, kernel)

# FREQUENCY DOMAIN (FOURIER)
# To multiply the image and kernel in Fourier, making them same size.
padded_kernel = np.zeros_like(original_image)

# Shift the kernel to center
padded_kernel[:kernel_size, :kernel_size] = kernel
start = kernel_size // 2
padded_kernel = np.roll(padded_kernel, -start, axis=0)
padded_kernel = np.roll(padded_kernel, -start, axis=1)

# Convert both Image and Kernel to Fourier Domain (FFT)
fft_image = np.fft.fft2(original_image)
fft_kernel = np.fft.fft2(padded_kernel)

# Perform Multiplication in Frequency Domain
fft_result = fft_image * fft_kernel

# Convert back to Spatial Domain (Inverse FFT)
fourier_blur_complex = np.fft.ifft2(fft_result)
fourier_blur = np.abs(fourier_blur_complex) # Take magnitude to remove imaginary noise


# PLOT 1: VISUALIZATION OF EACH STEP

plt.figure(figsize=(18, 12))
# Original Image
plt.subplot(2, 3, 1)
plt.imshow(original_image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

# Gaussian Kernel
plt.subplot(2, 3, 2)
plt.imshow(kernel, cmap='gray')
plt.title(f"Gaussian Kernel\n({kernel_size}x{kernel_size})")
plt.axis('off')

# Fourier of Image
plt.subplot(2, 3, 3)
plt.imshow(get_fourier_spectrum(original_image), cmap='inferno')
plt.title("Fourier of Image (Log Scale)")
plt.axis('off')

# Fourier of Kernel
plt.subplot(2, 3, 4)
plt.imshow(get_fourier_spectrum(padded_kernel), cmap='inferno')
plt.title("Fourier of Kernel (Log Scale)")
plt.axis('off')

# Result: Spatial Blur
plt.subplot(2, 3, 5)
plt.imshow(spatial_blur, cmap='gray')
plt.title("Result: Spatial Blur")
plt.axis('off')

# Result: Fourier Blur
plt.subplot(2, 3, 6)
plt.imshow(fourier_blur, cmap='gray')
plt.title("Result: Fourier Blur")
plt.axis('off')
plt.subplots_adjust(hspace=0.1, wspace=0.1)
plt.show()


# PLOT 2: DETAILED HISTOGRAM ANALYSIS BETWEEN TWO APPROACHES
plt.figure(figsize=(18, 5))

# Histogram of Spatial Filter Output
plt.subplot(1, 2, 1)
plt.hist(spatial_blur.ravel(), bins=100, color='blue', alpha=0.7)
plt.title("Histogram: Spatial Blur Result")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency (Log Scale)")
plt.yscale('log')
plt.grid(True, alpha=0.3)

# Histogram of Fourier Filter Output
plt.subplot(1, 2, 2)
plt.hist(fourier_blur.ravel(), bins=100, color='red', alpha=0.7)
plt.title("Histogram: Fourier Blur Result")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency (Log Scale)")
plt.yscale('log')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()