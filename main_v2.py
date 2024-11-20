import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import search
import lbp


def hsv_to_black(image, lower_hsv, upper_hsv):
    """
    Convert specific HSV color ranges to black in the image.
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
    image[mask > 0] = [0, 0, 0]  # Set matching pixels to black
    return image

def apply_morphology(image, kernel_size):
    """
    Apply morphological closing to the grayscale image.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def blur_and_sobel(image):
    blurred = cv2.GaussianBlur(image, (3, 3), 0)

    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = cv2.convertScaleAbs(sobelx)
    
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobely = cv2.convertScaleAbs(sobely)
    
    sobel_combined = cv2.addWeighted(abs_sobelx, 0.5, abs_sobely, 0.5, 0)
    
    return sobel_combined

def search_and_lbp(sobel_image):
    radius = 1
    n_points = 8
    noise_threshold = 1000 # 定義噪點區域大小的閾值
    search_img = search.bfs_remove_noise_optimized(sobel_image, noise_threshold)
    
    #lbp
    lbp_image = lbp.calculate_lbp(search_img, radius, n_points)
    return lbp_image

def mark_white_area_in_lbp(origin_image, lbp_image):
    """
    Mark the white areas in the bottom third of the LBP image on the original image in red.
    """
    image = origin_image.copy()
    # Get the dimensions of the LBP image
    height, width = lbp_image.shape
    
    # Define the bottom third region of the image
    bottom_third = int(height * 1 / 3)  # Bottom third starts from 2/3 of the image height
    
    # Extract the bottom third of the LBP image
    lbp_bottom = lbp_image[bottom_third:, :]
    
    # Find the white areas (255) in the bottom third of the LBP image
    white_area = lbp_bottom == 255
    
    # Mark the corresponding areas on the original image in red
    image[bottom_third:, :][white_area] = [0, 0, 255]  # Red color (BGR format)

    return image

    
def save_image(output_dir, filename, image):
    """
    Save an image to the specified output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, filename), image)



def main():
    # Parameters
    input_path = 'test4.jpg'
    output_dir = 'result'
    print(type(output_dir))
    hsv_lower = np.array([70, 0, 0])  # HSV lower bound
    hsv_upper = np.array([180, 255, 230])  # HSV upper bound
    kernel_size = 3  # Morphological kernel size

    # Load image
    image = cv2.imread(input_path)

    # HSV filtering
    hsv_img = hsv_to_black(image.copy(), hsv_lower, hsv_upper)

    # Convert to grayscale
    gray_image = cv2.cvtColor(hsv_img, cv2.COLOR_BGR2GRAY)

    # Apply morphological closing
    closed_image = apply_morphology(gray_image, kernel_size)
    
    sobel_image = blur_and_sobel(closed_image)
    
    lbp_image = search_and_lbp(sobel_image)

    colored_img = mark_white_area_in_lbp(image, lbp_image)


    # Save results
    save_image(output_dir, 'hsv.jpg', hsv_img)
    save_image(output_dir, 'image_gray.jpg', gray_image)
    save_image(output_dir, 'closed_image.jpg', closed_image)
    save_image(output_dir, 'sobel_image.jpg', sobel_image)
    save_image(output_dir, 'lbp_image.jpg', lbp_image)
    save_image(output_dir, 'colored_img.jpg', colored_img)
    
    # Display results (optional)
    cv2.imshow('Original Image', image)
    cv2.imshow('HSV Processed', hsv_img)
    cv2.imshow('Grayscale Image', gray_image)
    cv2.imshow('Morphological Closing', closed_image)
    cv2.imshow('sobel_image', sobel_image)
    cv2.imshow('lbp_image', lbp_image)
    cv2.imshow('colored_img', colored_img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
