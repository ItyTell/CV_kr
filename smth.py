import cv2
import numpy as np
from skimage.feature import hog
from skimage import exposure
import math

def detect_log_corners(image, sigma=2, threshold=0.1):
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    
    # Calculate Laplacian
    laplacian = cv2.Laplacian(blur, cv2.CV_64F)
    
    # Find local maxima
    kernel_size = 3
    local_max = cv2.dilate(laplacian, np.ones((kernel_size, kernel_size)))
    corner_peaks = (laplacian == local_max) & (laplacian > threshold)
    
    # Get corner coordinates
    corners = np.column_stack(np.where(corner_peaks))
    return corners

def extract_hog_features(image):
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # Calculate HOG features
    features, hog_image = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True,
        feature_vector=True
    )
    
    return features, hog_image

def match_keypoints(features1, features2):
    # Calculate similarity matrix using dot product
    similarity = np.dot(features1, features2.T)
    
    # Find best matches
    matches = []
    for i in range(len(features1)):
        best_match = np.argmax(similarity[i])
        if similarity[i][best_match] > 0.8:  # Threshold for good matches
            matches.append((i, best_match))
            
    return matches

def calculate_distance(known_distance, known_size, pixel_size):
    # Using similar triangles principle
    # known_distance/known_size = estimated_distance/pixel_size
    estimated_distance = (known_distance * pixel_size) / known_size
    return estimated_distance

def process_images(gt_image_path, target_images, known_distance, object_size):
    # Load ground truth image
    gt_image = cv2.imread(gt_image_path)
    
    # Process ground truth image
    gt_corners = detect_log_corners(gt_image)
    gt_features, _ = extract_hog_features(gt_image)
    
    estimated_distances = {}
    
    # Process each target image
    for name, img_path in target_images.items():
        target_image = cv2.imread(img_path)
        
        # Detect corners and extract features
        target_corners = detect_log_corners(target_image)
        target_features, _ = extract_hog_features(target_image)
        
        # Match keypoints
        matches = match_keypoints(gt_features, target_features)
        
        if len(matches) > 0:
            # Calculate average pixel size ratio between matched points
            pixel_ratios = []
            for match in matches:
                gt_idx, target_idx = match
                gt_point = gt_corners[gt_idx]
                target_point = target_corners[target_idx]
                
                # Calculate distances between consecutive points
                if gt_idx > 0 and target_idx > 0:
                    gt_dist = np.linalg.norm(gt_corners[gt_idx] - gt_corners[gt_idx-1])
                    target_dist = np.linalg.norm(target_corners[target_idx] - target_corners[target_idx-1])
                    if gt_dist > 0:
                        pixel_ratios.append(target_dist / gt_dist)
            
            if pixel_ratios:
                avg_pixel_ratio = np.mean(pixel_ratios)
                estimated_distance = calculate_distance(known_distance, object_size[0], avg_pixel_ratio * object_size[0])
                estimated_distances[name] = estimated_distance
            
    return estimated_distances

# Main execution
if __name__ == "__main__":
    # Configuration
    gt_image_path = "data/book/book_1.jpg"
    target_images = {
        "book_2.jpg": "data/book/book_2.jpg",
        "book_3.jpg": "data/book/book_3.jpg",
        "book_4.jpg": "data/book/book_4.jpg",
        "book_5.jpg": "data/book/book_5.jpg"
    }
    known_distance = 100  # cm
    object_size = (7.5, 25, 17.5)  # cm (height, width, depth)
    
    # Process images and calculate distances
    estimated_distances = process_images(gt_image_path, target_images, known_distance, object_size)
    
    # Calculate average error
    gt_values = {
        "book_2.jpg": 45,
        "book_3.jpg": 94,
        "book_4.jpg": 152,
        "book_5.jpg": 119
    }
    
    total_error = 0
    for name, est_dist in estimated_distances.items():
        error = abs(est_dist - gt_values[name])
        total_error += error
        print(f"Image: {name}")
        print(f"Estimated distance: {est_dist:.2f} cm")
        print(f"Ground truth distance: {gt_values[name]} cm")
        print(f"Error: {error:.2f} cm\n")
    
    average_error = total_error / len(estimated_distances)
    print(f"Average error: {average_error:.2f} cm")