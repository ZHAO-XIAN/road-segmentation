import cv2
import numpy as np
from skimage.feature import local_binary_pattern

class LBP:
    def __init__(self, radius=1, n_points=8, winSize=10):
        self.radius = radius
        self.n_points = n_points
        self.winSize = winSize
        self.lbp = None

    def setLBPImageScikit(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray_image, self.n_points, self.radius)
        lbp = np.array(lbp, dtype=np.uint8)
        self.lbp = lbp
        return lbp


def create_road_markers(image):
    # Convert to grayscale and apply edge detection (Canny)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_image = cv2.medianBlur(gray_image, 3)
    edges = cv2.Canny(blur_image, 0, 255)
    
    # Use the edges as foreground markers (road area)
    foreground_markers = np.zeros_like(blur_image)
    foreground_markers[edges > 0] = 1  # Set edge points as foreground
    
    # Use thresholding to define background (non-road area)
    _, background_markers = cv2.threshold(gray_image, 110, 255, cv2.THRESH_BINARY_INV)
    # background_markers = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    background_markers[background_markers == 255] =   # Mark as background (2)
    
    # Combine both foreground and background markers
    markers = foreground_markers + background_markers
    markers = markers.astype(np.int32)
    
    return markers, edges, background_markers, gray_image


def apply_watershed(image, markers):
    # Convert markers to the format required by cv2.watershed
    markers = markers + 1  # Increment marker values to separate background and foreground
    markers[markers == 1] = 0  # Unknown region

    # Apply the Watershed algorithm
    markers = cv2.watershed(image, markers)
    
    # Mark boundaries in bright red and make them more obvious
    image[markers == -1] = [0, 0, 255]  # Bright red boundary

    return image


if __name__ == '__main__':
    winSize = 10
    lbp = LBP(winSize=winSize)

    # Read the image
    image = cv2.imread('test4.jpg')
    image = cv2.resize(image, (360,480), interpolation=cv2.INTER_AREA)
    
    # Generate LBP image (optional for visualization)
    lbp_image = lbp.setLBPImageScikit(image)

    # Create road markers using edge detection and thresholding
    markers, canny, background_markers, gray_image = create_road_markers(image)

    # Apply the Watershed algorithm using refined markers
    segmented_image = apply_watershed(image, markers)

    # Display results
    cv2.imshow("LBP Image", lbp_image)
    cv2.imshow("Markers", markers)  # Show markers (scaled for visibility)
    cv2.imshow("Canny", canny)  # Show markers (scaled for visibility)
    cv2.imshow("gray_image", gray_image)  # Show markers (scaled for visibility)
    cv2.imshow("background_markers", background_markers)
    cv2.imshow("Segmented Image", segmented_image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
