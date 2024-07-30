import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def calculate_reflection_metric_hsv(image, v_threshold=250, s_threshold=5):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv_image)
    
    # Apply threshold to find highly bright regions in the Value channel
    _, bright_regions = cv2.threshold(v_channel, v_threshold, 255, cv2.THRESH_BINARY)

    # Apply inverse threshold to exclude highly saturated regions
    _, low_saturation_regions = cv2.threshold(s_channel, s_threshold, 255, cv2.THRESH_BINARY_INV)

    # Combine the bright regions with low saturation to isolate reflections
    reflection_mask = cv2.bitwise_and(bright_regions, low_saturation_regions)
    
    # Calculate the reflection metric as the proportion of bright, low-saturation pixels
    total_pixels = image.shape[0] * image.shape[1]
    reflection_pixels = np.count_nonzero(reflection_mask)
    reflection_metric = reflection_pixels / total_pixels
    
    return reflection_metric, reflection_mask

def save_reflection_overlay(original_image, reflection_mask, output_folder, filename):
    # Create an overlay where the reflection mask is applied in red
    overlay_image = original_image.copy()
    overlay_image[reflection_mask > 0, 0] = 0
    overlay_image[reflection_mask > 0, 1] = 0
    overlay_image[reflection_mask > 0, 2] = 255
    
    # Save the overlay image
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, overlay_image)

def analyze_and_visualize_folder(input_folder, output_folder, v_threshold=200, s_threshold=50, visualization_interval=30):
    metrics = []
    images_with_reflections = 0

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process all images in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            
            # Calculate reflection metric using HSV color space
            metric, reflection_mask = calculate_reflection_metric_hsv(image, v_threshold, s_threshold)
            metrics.append(metric)

            if metric > 0:
                images_with_reflections += 1
                if images_with_reflections % visualization_interval == 0:
                    save_reflection_overlay(image, reflection_mask, output_folder, f"overlay_{filename}")

    # Calculate and print out the statistics
    if metrics:
        average_metric = np.mean(metrics)
        reflection_metrics = [m for m in metrics if m > 0]
        mean_reflection_metric = np.mean(reflection_metrics) if reflection_metrics else 0
        print(f"Average Reflection Metric (All Images): {average_metric:.4f}")
        print(f"Mean Reflection Metric for Images with Reflections: {mean_reflection_metric:.4f}")
        print(f"Number of Images with Reflections: {images_with_reflections}")
        print(f"Total Number of Analyzed Images: {len(metrics)}")
    else:
        print("No images found or analyzed.")

# Specify the input and output folder paths
folder_path = '/media/niclas/T7/reflection_final/odometry/iphone/rgb'
output_folder_path = 'reflection_masks_hsv'
analyze_and_visualize_folder(folder_path, output_folder_path)