# Optical Flow and Structure from Motion

## Project Components

### 1. Optical Flow Estimation
* **Goal:** Calculate and visualize the apparent motion of pixels between consecutive video frames.
* **Method:** Implements motion vector tracking to determine the direction and magnitude of pixel displacement.

### 2. Planar Structure from Motion
* **Goal:** Reconstruct the 3D geometry of a flat object and the camera's path from a 2D video sequence.
* **Method:** Employs **SIFT** feature matching and **Homography Decomposition** to map 2D pixels to a defined $Z=0$ world plane.
* **Output:** A visualization showing the object boundary and the calculated camera poses ($R$ and $t$).

---

## Prerequisites
Install the necessary libraries using:

```bash
pip install opencv-python numpy matplotlib