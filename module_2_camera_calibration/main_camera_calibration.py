import cv2
import numpy as np
import glob
import os

# --- CONFIGURATION ---
CHECKERBOARD_DIMS = (9, 6) 
SQUARE_SIZE = 0.037  # Meters
CALIB_IMG_DIR = 'resources/calibration_images'
TEST_IMAGE_PATH = 'resources/Test_1.JPG'
TEST_DISTANCE_CAMERA_TO_OBJECT = 2.6  # Meters

# Global variables
click_points = []

def run_calibration():
    # Create a 3D grid of object points (0,0,0), (1,0,0), ...
    objp = np.zeros((CHECKERBOARD_DIMS[0] * CHECKERBOARD_DIMS[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_DIMS[0], 0:CHECKERBOARD_DIMS[1]].T.reshape(-1, 2)
    objp = objp * SQUARE_SIZE

    objpoints = [] 
    imgpoints = [] 

    images = glob.glob(f'{CALIB_IMG_DIR}/*.JPG')
    
    if not images:
        print("Error: No images found")
        return None, None

    print(f"Processing {len(images)} images...")

    valid_images = 0
    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect checkerboard corners in the image
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_DIMS, None)

        if ret == True:
            valid_images += 1
            objpoints.append(objp)
            
            # Refine corner locations to sub-pixel accuracy for better calibration
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)
        else:
            print(f"  [SKIP] Pattern not found: {os.path.basename(fname)}")

    # Compute camera matrix and distortion coefficients
    if len(objpoints) > 0:
        print(f"\nCalibrating with {valid_images} valid images...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None) # type: ignore
        print(f"Calibration Complete. Reprojection Error (RMS): {ret:.4f} pixels\n")
        return mtx, dist
    else:
        print("Calibration failed. No valid images found.")
        return None, None

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Capture up to 4 points
        if len(click_points) < 4:
            click_points.append((x, y))
            
            # Draw a visual marker on the image for feedback
            cv2.circle(param, (x, y), 10, (0, 0, 255), -1) 
            
            # Draw lines connecting the points to visualize the box
            if len(click_points) > 1:
                cv2.line(param, click_points[-2], click_points[-1], (0, 255, 0), 2)
            if len(click_points) == 4:
                # Close the box by connecting the last point to the first
                cv2.line(param, click_points[3], click_points[0], (0, 255, 0), 2)

            cv2.imshow("Measure Object", param)

def measure_object(image_path, distance_Z, K, D):
    global click_points
    click_points = [] 
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Image {image_path} not found.")
        return

    # Display the original raw image to avoid distortion artifacts during selection
    cv2.namedWindow("Measure Object", cv2.WINDOW_NORMAL)
    cv2.imshow("Measure Object", img)
    cv2.setMouseCallback("Measure Object", click_event, img)
    
    print("INSTRUCTIONS: Click the 4 corners of the object (Order: TL -> TR -> BR -> BL)")

    # Wait until 4 points are selected
    while True:
        if len(click_points) == 4:
            cv2.waitKey(1000) 
            break
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    if len(click_points) == 4:
        # Reshape input points to (N, 1, 2) format required by OpenCV functions
        points_raw = np.array(click_points, dtype=np.float32).reshape(-1, 1, 2)
        
        # Mathematically undistort only the selected point coordinates
        points_undistorted = cv2.undistortPoints(points_raw, K, D, P=K)
        
        # Flatten the output array back to a simple list of (x, y) coordinates
        pts = points_undistorted.reshape(-1, 2)
        
        # Calculate pixel distances for Width (Top/Bottom) and Height (Left/Right)
        w1 = np.linalg.norm(pts[0] - pts[1])
        w2 = np.linalg.norm(pts[3] - pts[2])
        avg_w_pix = (w1 + w2) / 2
        
        h1 = np.linalg.norm(pts[0] - pts[3])
        h2 = np.linalg.norm(pts[1] - pts[2])
        avg_h_pix = (h1 + h2) / 2
        
        # Extract focal length from the camera matrix
        f_pixels = (K[0, 0] + K[1, 1]) / 2
        
        # Apply perspective projection equation to convert pixels to real-world meters
        real_w = (avg_w_pix * distance_Z) / f_pixels
        real_h = (avg_h_pix * distance_Z) / f_pixels
        
        print(f"\n--- RESULTS ---")
        print(f"Width:  {real_w:.4f} m ({real_w * 100:.2f} cm)")
        print(f"Height: {real_h:.4f} m ({real_h * 100:.2f} cm)")
        
if __name__ == "__main__":
    camera_matrix, dist_coeffs = run_calibration()
    if camera_matrix is not None:

        # Real dimension of the object in Test_1.JPG is size: 13.4 x 19.9 cm
        measure_object(TEST_IMAGE_PATH, TEST_DISTANCE_CAMERA_TO_OBJECT, camera_matrix, dist_coeffs)