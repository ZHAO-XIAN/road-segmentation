import cv2
import numpy as np

def adjust_image(val):
    # Read the image
    image = cv2.imread('test4.jpg')
    
    # Get values from trackbars
    lower_h = cv2.getTrackbarPos('Lower H', 'Adjust HSV')
    lower_s = cv2.getTrackbarPos('Lower S', 'Adjust HSV')
    lower_v = cv2.getTrackbarPos('Lower V', 'Adjust HSV')
    upper_h = cv2.getTrackbarPos('Upper H', 'Adjust HSV')
    upper_s = cv2.getTrackbarPos('Upper S', 'Adjust HSV')
    upper_v = cv2.getTrackbarPos('Upper V', 'Adjust HSV')
    
    # Create HSV lower and upper bounds
    lower = np.array([lower_h, lower_s, lower_v])
    upper = np.array([upper_h, upper_s, upper_v])
    
    # Convert image to HSV and apply the mask
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, lower, upper)
    result = cv2.bitwise_and(image, image, mask=mask)
    
    # Convert to grayscale and apply morphology
    image_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(image_gray, cv2.MORPH_CLOSE, kernel)
    
    # Show the result
    cv2.imshow('Result', closing)

def main():
    # Create a window
    cv2.namedWindow('Adjust HSV')

    # Create trackbars for HSV lower and upper bounds
    cv2.createTrackbar('Lower H', 'Adjust HSV', 80, 180, adjust_image)
    cv2.createTrackbar('Lower S', 'Adjust HSV', 30, 255, adjust_image)
    cv2.createTrackbar('Lower V', 'Adjust HSV', 70, 255, adjust_image)
    cv2.createTrackbar('Upper H', 'Adjust HSV', 120, 180, adjust_image)
    cv2.createTrackbar('Upper S', 'Adjust HSV', 70, 255, adjust_image)
    cv2.createTrackbar('Upper V', 'Adjust HSV', 130, 255, adjust_image)

    # Call the function once to initialize
    adjust_image(0)
    
    # Wait until the user presses a key
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
