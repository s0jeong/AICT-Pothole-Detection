import cv2
import numpy as np
import random
import os
from pathlib import Path

class PotholeSynthesizer:
    def __init__(self, road_image_path, mask_path, pothole_folder, output_folder, label_folder):
        # Initialize paths for road image, mask, pothole image folder, output folder, and label folder
        self.road_image_path = road_image_path
        self.mask_path = mask_path
        self.pothole_folder = pothole_folder
        self.output_folder = output_folder
        self.label_folder = label_folder
        
        # Create output and label folders (if they don't exist)
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        Path(label_folder).mkdir(parents=True, exist_ok=True)
        
        # Load road image
        self.road_image = cv2.imread(road_image_path)
        if self.road_image is None:
            raise ValueError(f"Could not load road image: {road_image_path}")
        # Load mask image (grayscale)
        self.road_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if self.road_mask is None:
            raise ValueError(f"Could not load mask image: {mask_path}")
        # Binarize mask (255 for values >= 127, 0 otherwise)
        self.road_mask = cv2.threshold(self.road_mask, 127, 255, cv2.THRESH_BINARY)[1]
        
        # Load pothole images
        self.pothole_images = self.load_pothole_images()
        
        # Set randomization parameters
        self.randomization_params = {
            'size_range': (60, 90),  # Pothole size range (pixels)
            'brightness_range': (0.3, 0.7),  # Brightness adjustment range
            'pothole_count': random.randint(1, 3)  # Randomly synthesize 1 to 3 potholes per image
        }
    
    def load_pothole_images(self):
        # Load pothole images
        pothole_images = []
        # Process supported image extensions
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            pothole_files = Path(self.pothole_folder).glob(ext)
            for file_path in pothole_files:
                # Load image (including alpha channel)
                img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
                if img is None:
                    print(f"Failed to load pothole image: {file_path}")
                    continue
                # Convert 3-channel image to 4-channel (BGRA)
                if img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                pothole_images.append(img)
        if not pothole_images:
            print("Warning: No pothole images found.")
        return pothole_images
    
    def apply_brightness_adjustment(self, img, factor):
        # Apply brightness adjustment
        # Convert to HSV color space
        hsv = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2HSV).astype(np.float32)
        # Adjust Value (V) channel
        hsv[:, :, 2] = hsv[:, :, 2] * factor
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)  # Clamp values to 0-255 range
        # Convert back to BGR
        bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        result = np.zeros_like(img)
        result[:, :, :3] = bgr
        result[:, :, 3] = img[:, :, 3]  # Maintain alpha channel
        return result
    
    def resize_pothole(self, pothole_img, target_size=None):
        # Resize pothole
        if target_size is None:
            target_size = random.randint(*self.randomization_params['size_range'])
        h, w = pothole_img.shape[:2]
        max_dim = max(h, w)
        scale = target_size / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(pothole_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    def rotate_image(self, img, angle=None):
        # Image rotation (not used, returns original)
        return img
    
    def overlay_image_alpha(self, img, img_overlay, x, y):
        # Image blending with alpha channel (with subtle feathering)
        overlay_h, overlay_w = img_overlay.shape[:2]
        if x + overlay_w > img.shape[1] or y + overlay_h > img.shape[0] or x < 0 or y < 0:
            return img, None  # Ignore if out of bounds, return None
        if img_overlay.shape[2] == 4:
            bgr = img_overlay[:, :, :3]
            alpha = img_overlay[:, :, 3] / 255.0
        else:
            bgr = img_overlay
            alpha = np.ones((overlay_h, overlay_w), dtype=np.float32)
        # Apply subtle feathering
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
        alpha = alpha[..., np.newaxis]
        roi = img[y:y+overlay_h, x:x+overlay_w].astype(np.float32)
        blended = alpha * bgr.astype(np.float32) + (1 - alpha) * roi
        img[y:y+overlay_h, x:x+overlay_w] = blended.astype(np.uint8)
        # Return the bounding box of the composited pothole
        return img, (x, y, x + overlay_w, y + overlay_h)
    
    def get_random_valid_position(self, pothole_size):
        # Return a random valid position within the mask
        h, w = self.road_mask.shape
        pothole_h, pothole_w = pothole_size
        valid_positions = np.where(self.road_mask == 255)  # Road area (255) positions
        if len(valid_positions[0]) == 0:
            raise ValueError("No valid mask area found.")
        while True:
            idx = random.randint(0, len(valid_positions[0]) - 1)
            y_center = valid_positions[0][idx]
            x_center = valid_positions[1][idx]
            x_start = max(0, x_center - pothole_w // 2)
            y_start = max(0, y_center - pothole_h // 2)
            if (x_start + pothole_w <= w and y_start + pothole_h <= h and
                np.all(self.road_mask[y_start:y_start + pothole_h, x_start:x_start + pothole_w] == 255)):
                return x_start, y_start
    
    def generate_synthetic_image(self, pothole_subset):
        # Generate synthetic pothole image and YOLO labels
        if not pothole_subset:
            raise ValueError("Pothole image list is empty.")
        result_img = self.road_image.copy()
        occupied_areas = []
        annotations = []  # Store YOLO-format annotations
        random.shuffle(pothole_subset)  # Shuffle pothole images
        pothole_subset = pothole_subset[:self.randomization_params['pothole_count']]  # Select 1-3
        
        for pothole_img in pothole_subset:
            pothole_img = pothole_img.copy()
            pothole_img = self.resize_pothole(pothole_img)  # Resize
            brightness_factor = random.uniform(*self.randomization_params['brightness_range'])
            pothole_img = self.apply_brightness_adjustment(pothole_img, brightness_factor)  # Adjust brightness
            max_attempts = 50
            for _ in range(max_attempts):
                x_start, y_start = self.get_random_valid_position(pothole_img.shape[:2])
                new_area = (x_start, y_start, x_start + pothole_img.shape[1], y_start + pothole_img.shape[0])
                overlap = False
                for occupied in occupied_areas:
                    if (new_area[0] < occupied[2] and new_area[2] > occupied[0] and
                        new_area[1] < occupied[3] and new_area[3] > occupied[1]):
                        overlap = True
                        break
                if not overlap:
                    result_img, bbox = self.overlay_image_alpha(result_img, pothole_img, x_start, y_start)  # Synthesize pothole
                    if bbox is not None:  # If synthesis was valid
                        occupied_areas.append(new_area)
                        # Add YOLO-format annotations (class_id, x_center, y_center, width, height)
                        img_h, img_w = result_img.shape[:2]
                        x_min, y_min, x_max, y_max = bbox
                        x_center = (x_min + x_max) / 2 / img_w
                        y_center = (y_min + y_max) / 2 / img_h
                        width = (x_max - x_min) / img_w
                        height = (y_max - y_min) / img_h
                        annotations.append((0, x_center, y_center, width, height))  # class_id=0 (pothole)
                    break
        return result_img, annotations
    
    def save_image_and_labels(self, result_img, annotations, output_name="pothole_synthetic"):
        # Save synthetic image and YOLO labels
        output_path = os.path.join(self.output_folder, f"{output_name}.jpg")
        cv2.imwrite(output_path, result_img)
        print(f"Synthetic image saved: {output_path}")
        
        # Save YOLO label file
        label_path = os.path.join(self.label_folder, f"{output_name}.txt")
        with open(label_path, 'w') as f:
            for ann in annotations:
                class_id, x_center, y_center, width, height = ann
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        print(f"YOLO labels saved: {label_path}")
        return output_path

if __name__ == "__main__":
    # Directory path setup
    total_data = ""
    mask_images = ""
    pothole_folder = ""
    output_folder = ""
    label_folder = ""
    
    # Get list of road images
    road_images = [f for f in os.listdir(total_data) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not road_images:
        print("Error: No image files found in the total_data directory.")
        exit()
    
    # Select all road images
    print(f"Processing {len(road_images)} road images")
    
    # Track the number of successfully generated images
    generated_count = 0
    
    # Process all road images
    for road_filename in road_images:
        road_image_path = os.path.join(total_data, road_filename)
        # Create corresponding mask filename
        file_name, _ = os.path.splitext(road_filename)
        mask_filename = f"{file_name}_mask.png"
        mask_path = os.path.join(mask_images, mask_filename)
        
        # Check if mask file exists
        if not os.path.exists(mask_path):
            print(f"Could not find mask for {road_filename}: {mask_path}")
            continue
        
        try:
            # Initialize PotholeSynthesizer
            synthesizer = PotholeSynthesizer(road_image_path, mask_path, pothole_folder, output_folder, label_folder)
            # Select pothole images
            pothole_subset = synthesizer.pothole_images
            if not pothole_subset:
                print(f"Error while processing {road_filename}: No pothole images found.")
                continue
            # Generate synthetic image and annotations
            result_img, annotations = synthesizer.generate_synthetic_image(pothole_subset)
            # Save image and labels with a unique name
            synthesizer.save_image_and_labels(result_img, annotations, output_name=f"synthetic_{file_name}")
            generated_count += 1
        except Exception as e:
            print(f"Error occurred while processing {road_filename}: {str(e)}")
    
    # Print completion message
    print(f"Generation complete: {generated_count} images")