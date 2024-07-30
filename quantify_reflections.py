import cv2
import numpy as np
import os

def calculate_reflection_metric(image, brightness_threshold=250, min_area=20):
    # Convert to grayscale if the image is in color
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Apply threshold to find bright regions
    _, bright_regions = cv2.threshold(gray_image, brightness_threshold, 255, cv2.THRESH_BINARY)
    
    # Apply morphological opening to remove small bright spots
    kernel = np.ones((3,3), np.uint8)
    bright_regions = cv2.morphologyEx(bright_regions, cv2.MORPH_OPEN, kernel)
    
    # Find contours in the bright regions mask
    contours, _ = cv2.findContours(bright_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out small contours
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            cv2.drawContours(bright_regions, [cnt], -1, (0,0,0), thickness=cv2.FILLED)
    
    # Calculate the reflection metric
    total_pixels = gray_image.size
    bright_pixels = np.count_nonzero(bright_regions)
    reflection_metric = bright_pixels / total_pixels
    
    reflections_exceed_1_percent = reflection_metric > 0.01
    
    return reflection_metric, bright_regions, reflections_exceed_1_percent



import matplotlib.pyplot as plt

def visualize_reflections(original_image, reflection_mask, i):
    # Create a copy of the original image to modify
    overlay_image = original_image.copy()

    # Where the mask is active, set the red channel to maximum (255)
    overlay_image[reflection_mask > 0, 0] = 0  # Set blue channel to 0 where mask is active
    overlay_image[reflection_mask > 0, 1] = 0  # Set green channel to 0 where mask is active
    overlay_image[reflection_mask > 0, 2] = 255  # Set red channel to 255 where mask is active

    # Plotting
    plt.figure(figsize=(10, 5))

    # Plot the original image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    # Plot the image with the red overlay
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
    plt.title('Reflections Overlay in Red')
    plt.axis('off')

    plt.tight_layout()
    #plt.show()

    # Save the overlay image
    cv2.imwrite(f"reflection_masks/{i}.jpg", overlay_image)




def analyze_folder(folder_path, folder_path2,brightness_threshold=250, visualization_interval=30):
    metrics = []
    images_with_reflections = 0
    visualized_count = 0
    images_exceeding_1_percent_reflection = 0  # Counter for images exceeding the 1% threshold

    i=0

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            
            # Calculate the reflection metric for the current image
            metric, reflection_mask, exceeds_1_percent = calculate_reflection_metric(image, brightness_threshold)
            metrics.append(metric)
            
            if metric > 0:  # At least one pixel exceeds the brightness threshold
                images_with_reflections += 1
                if exceeds_1_percent:
                    images_exceeding_1_percent_reflection += 1
                
                # Visualize every 30th image with reflections
                if  images_with_reflections % visualization_interval == 0:
                    i+=1
                    visualized_count += 1
                    print(f"Visualizing {filename} - Image {visualized_count} with Reflections")
                    visualize_reflections(image, reflection_mask, i)

    for filename in os.listdir(folder_path2):
        
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path2, filename)
            image = cv2.imread(image_path)
            
            
            # Calculate the reflection metric for the current image
            try:
                metric, reflection_mask, exceeds_1_percent = calculate_reflection_metric(image, brightness_threshold)
            except:
                continue
            metrics.append(metric)
            
            if metric > 0:  # At least one pixel exceeds the brightness threshold
                images_with_reflections += 1
                if exceeds_1_percent:
                    images_exceeding_1_percent_reflection += 1
                
                # Visualize every 30th image with reflections
                if  images_with_reflections % visualization_interval == 0:
                    i+=1
                    visualized_count += 1
                    print(f"Visualizing {filename} - Image {visualized_count} with Reflections")
                    visualize_reflections(image, reflection_mask, i)

    # Compute summary statistics
    if metrics:
        average_metric = np.mean(metrics)
        reflection_metrics = [m for m in metrics if m > 0]
        if reflection_metrics:
            mean_reflection_metric = np.mean(reflection_metrics)
            print(f"Mean Reflection Metric for Images with Reflections: {mean_reflection_metric:.4f}")
        else:
            print("No images with reflections detected.")
        
        print(f"Average Reflection Metric (All Images): {average_metric:.4f}")
        print(f"Number of Images with Reflections: {images_with_reflections}")
        print(f"Total Number of Analyzed Images: {len(metrics)}")
        print(f"Number of Images where reflections exceed 1%: {images_exceeding_1_percent_reflection}")

    else:
        print("No images found or analyzed.")

i=0
# Specify the path to your folder of images
folder_path = '/media/niclas/T7/reflection_final/odometry/iphone/rgb'
folder_path2 = '/media/niclas/T7/reflection_final/odometry/iphone/rgb_2'
analyze_folder(folder_path, folder_path2)
