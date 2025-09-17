import cv2
import numpy as np
import os

# Input and output directories
input_dir = ''
save_dir = ''

# Create output directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Supported image extensions
image_extensions = ('.jpg', '.jpeg', '.png')

# Iterate through all files in the input directory
for filename in os.listdir(input_dir):
    # Check if the file is an image
    if filename.lower().endswith(image_extensions):
        # Full path to the image
        image_path = os.path.join(input_dir, filename)
        
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {filename}")
            continue

        # HSV conversion
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Grayscale mask (color difference + brightness)
        b, g, r = cv2.split(image)
        diff_rg = cv2.absdiff(r, g)
        diff_rb = cv2.absdiff(r, b)
        diff_gb = cv2.absdiff(g, b)

        threshold = 10
        mask_gray = (diff_rg < threshold) & (diff_rb < threshold) & (diff_gb < threshold)

        brightness = (r.astype(np.int32) + g.astype(np.int32) + b.astype(np.int32)) // 3
        mask_brightness = (brightness > 50) & (brightness < 200)

        mask_gray_brightness = mask_gray & mask_brightness

        # HSV saturation and value conditions
        mask_s = (s < 60)  # Low saturation for roads
        mask_v = (v > 40) & (v < 180)  # Exclude too dark or too bright areas

        mask_hsv = mask_s & mask_v

        # Final mask
        final_mask = mask_gray_brightness & mask_hsv
        final_mask = final_mask.astype(np.uint8) * 255

        # Save the mask
        file_name, _ = os.path.splitext(filename)
        mask_file_name = f"{file_name}_mask.png"
        save_path = os.path.join(save_dir, mask_file_name)
        
        cv2.imwrite(save_path, final_mask)
        print(f"Saved mask for {filename} at {save_path}")