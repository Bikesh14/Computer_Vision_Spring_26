# Thermal Image Boundary Detection

This project compares two methods for finding exact animal boundaries in thermal imaging:
1. **Traditional CV:** HSV Thresholding + Morphological Closing + Canny Edge Detection.
2. **Deep Learning:** Meta's [Segment Anything Model 2 (SAM2)](https://github.com/facebookresearch/segment-anything-2).

## Prerequisites
Install the required libraries (including SAM2 from source):

```bash
pip install opencv-python numpy matplotlib torch
pip install git+[https://github.com/facebookresearch/segment-anything-2.git](https://github.com/facebookresearch/segment-anything-2.git)