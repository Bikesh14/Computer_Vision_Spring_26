# Spatial vs. Frequency Domain Blurring

This project demonstrates the **Convolution Theorem** by implementing image blurring in two ways:
1. **Spatial Domain:** Standard convolution (`cv2.filter2D`).
2. **Frequency Domain:** Multiplication in the Fourier domain ($F(u,v) \cdot H(u,v)$).

## Prerequisites
Install the required libraries:

```bash
pip install opencv-python numpy matplotlib