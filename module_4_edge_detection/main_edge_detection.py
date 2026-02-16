import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# CONFIGURATION 
IMAGE_PATH = '/Users/bbimali1/Documents/Computer_Vision_Spring_26/module_4_edge_detection/images/thermal_image_dog.jpg'
SAM2_CHECKPOINT = "/Users/bbimali1/Documents/Computer_Vision_Spring_26/module_4_edge_detection/sam2_checkpoints/sam2_hiera_tiny.pt" 
SAM2_CONFIG = "sam2_hiera_t.yaml"

def get_boundary_cv(img_bgr):

    # Convert to HSV Color Space
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # Hot colors range (Red, Orange, Yellow)
    lower1 = np.array([0, 50, 50]); upper1 = np.array([40, 255, 255])
    lower2 = np.array([170, 50, 50]); upper2 = np.array([180, 255, 255])
    
    # Combine ranges
    mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))
    
    # Clean up holes with Morphology
    kernel = np.ones((15,15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Extract the Boundary (Edge)
    edges = cv2.Canny(mask, 100, 200)
    
    return edges, mask

def get_boundary_sam2(img_bgr):

    # Setup SAM2
    if not os.path.exists(SAM2_CHECKPOINT):
        print(f"Error: Missing {SAM2_CHECKPOINT}")
        return np.zeros(img_bgr.shape[:2], dtype=np.uint8)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading SAM2 on {device}...")
    
    sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
    mask_generator = SAM2AutomaticMaskGenerator(sam2_model)

    # Generate Segmentation
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(img_rgb)

    if len(masks) == 0:
        return np.zeros(img_bgr.shape[:2], dtype=np.uint8)

    # Select the correct mask
    # Pick the largest mask that isn't the entire background
    best_mask = None
    max_area = 0
    total_area = img_bgr.shape[0] * img_bgr.shape[1]

    for m in masks:
        area = m['area']
        # Filter: Must be big enough, but not >90% of image (background)
        if area > max_area and area < (0.9 * total_area):
            max_area = area
            best_mask = m['segmentation']

    if best_mask is None:
        print("SAM2 could not isolate the object.")
        return np.zeros(img_bgr.shape[:2], dtype=np.uint8)

    # Convert Segmentation to Boundary
    binary_mask = best_mask.astype(np.uint8) * 255
    edges = cv2.Canny(binary_mask, 100, 200)
    
    return edges

def main():
    # Load Image
    if not os.path.exists(IMAGE_PATH):
        print("Error: Image not found.")
        return
    img = cv2.imread(IMAGE_PATH)

    print("Running OpenCV Method...")
    edges_cv, _ = get_boundary_cv(img)
    print("Running SAM2 Method...")
    edges_sam = get_boundary_sam2(img)

    h, w = edges_cv.shape
    comparison = np.zeros((h, w, 3), dtype=np.uint8)

    # Create Boolean Masks for each condition
    # Match: Pixel is >0 in BOTH images
    match_mask = (edges_cv > 0) & (edges_sam > 0)
    
    # CV Only: Pixel is >0 in CV but 0 in SAM
    cv_only_mask = (edges_cv > 0) & (edges_sam == 0)
    
    # SAM Only: Pixel is >0 in SAM but 0 in CV
    sam_only_mask = (edges_sam > 0) & (edges_cv == 0)

    # Assign Colors (RGB format for Matplotlib)
    comparison[match_mask] = [255, 0, 0]  
    comparison[cv_only_mask] = [0, 255, 0] 
    comparison[sam_only_mask] = [0, 0, 255] 

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # type: ignore
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(edges_cv, cmap='gray')
    plt.title("Method 1: OpenCV")
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(edges_sam, cmap='gray')
    plt.title("Method 2: SAM2")
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(comparison)
    plt.title("Comparison\n(Red=Match, Green=CV, Blue=SAM)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()