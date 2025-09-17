import os
import json
import cv2
import numpy as np
from tqdm import tqdm
import shutil

def change_asphalt_color(image, pothole_polygons, color_type="random"):
    """
    Changes the color of the asphalt area using the provided BGR+HSV logic, excluding pothole regions.
    """
    # 1. Create a mask for pothole areas
    pothole_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for seg in pothole_polygons:
        polys = seg if isinstance(seg[0], list) else [seg]
        for poly in polys:
            pts = np.array(poly, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(pothole_mask, [pts], 255)

    # <--- Start of integrating the provided successful logic ---
    # 2. Separate BGR and HSV channels
    b, g, r = cv2.split(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 3. Create a mask for BGR color difference and brightness
    threshold = 10
    mask_gray = (cv2.absdiff(r, g) < threshold) & \
                (cv2.absdiff(r, b) < threshold) & \
                (cv2.absdiff(g, b) < threshold)

    brightness = (r.astype(np.int32) + g.astype(np.int32) + b.astype(np.int32)) // 3
    mask_brightness = (brightness > 50) & (brightness < 200)
    
    mask_gray_brightness = mask_gray & mask_brightness

    # 4. Create a mask for HSV saturation and value
    mask_s = (s < 60)
    mask_v = (v > 40) & (v < 180)
    mask_hsv = mask_s & mask_v

    # 5. Create the final asphalt mask (satisfying both conditions)
    asphalt_mask = (mask_gray_brightness & mask_hsv).astype(np.uint8) * 255
    
    # Exclude pothole areas from the asphalt mask
    asphalt_mask = cv2.bitwise_and(asphalt_mask, cv2.bitwise_not(pothole_mask))
    # <--- End of integrating the provided successful logic ---

    if np.count_nonzero(asphalt_mask) == 0:
        return image.copy()  # Return original if no asphalt area is found

    # 6. Choose a color
    if color_type == "random":
        options = ["red", "white", "yellow", "ochre"]
        color_type = np.random.choice(options)
    
    hue_value, saturation_factor, value_factor = {
        "red":     (5, 50.0, 1.0),
        "white":   (0, 50.0, 1.5),
        "yellow":  (30, 50.0, 1.1),
        "ochre":   (20, 50.0, 1.0)
    }.get(color_type, ("red", 50.0, 1.0))

    # 7. Change the color of only the asphalt area
    hsv_aug = hsv.copy()
    hsv_aug[asphalt_mask == 255, 0] = hue_value
    hsv_aug[asphalt_mask == 255, 1] = np.clip(hsv_aug[asphalt_mask == 255, 1] * saturation_factor, 0, 255)
    hsv_aug[asphalt_mask == 255, 2] = np.clip(hsv_aug[asphalt_mask == 255, 2] * value_factor, 0, 255)

    colored_image = cv2.cvtColor(hsv_aug, cv2.COLOR_HSV2BGR)

    # 8. Image composition (modified for a more concise and safe approach)
    final_image = image.copy()
    # Replace the pixels in the asphalt_mask area of the original image with the new colored pixels
    final_image[asphalt_mask == 255] = colored_image[asphalt_mask == 255]
    
    return final_image

def augment_dataset(base_dir):
    print(f"Starting data augmentation for the path '{base_dir}'.")
    
    annotations_path = os.path.join(base_dir, "annotations_sam.json")
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)

    # <--- Fix path bug ---
    augmented_image_dir = os.path.join(base_dir, "images_augmented")
    if os.path.exists(augmented_image_dir):
        shutil.rmtree(augmented_image_dir)
    os.makedirs(augmented_image_dir, exist_ok=True)
    
    new_coco_data = coco_data.copy()
    new_coco_data['images'] = list(coco_data['images'])
    new_coco_data['annotations'] = list(coco_data['annotations'])

    max_img_id = max((img['id'] for img in coco_data['images']), default=0)
    max_ann_id = max((ann['id'] for ann in coco_data['annotations']), default=0)

    images_by_id = {img['id']: img for img in coco_data['images']}
    annotations_by_img_id = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        annotations_by_img_id.setdefault(img_id, []).append(ann)

    for img_id, img_info in tqdm(images_by_id.items(), desc="Augmenting images"):
        if img_id not in annotations_by_img_id:
            continue

        img_path = os.path.join(base_dir, "images", img_info['file_name'])
        image = cv2.imread(img_path)
        if image is None:
            continue

        img_annotations = annotations_by_img_id[img_id]
        polygons = [ann['segmentation'] for ann in img_annotations]

        augmented_image = change_asphalt_color(image, polygons)

        base_name, ext = os.path.splitext(img_info['file_name'])
        new_filename = f"{base_name}_aug{ext}"
        cv2.imwrite(os.path.join(augmented_image_dir, new_filename), augmented_image)

        max_img_id += 1
        new_img_info = img_info.copy()
        new_img_info['id'] = max_img_id
        # <--- Correctly set the image path in the JSON file ---
        new_img_info['file_name'] = os.path.join("images_augmented", new_filename)
        new_coco_data['images'].append(new_img_info)

        for ann in img_annotations:
            max_ann_id += 1
            new_ann = ann.copy()
            new_ann['id'] = max_ann_id
            new_ann['image_id'] = max_img_id
            new_coco_data['annotations'].append(new_ann)

    output_path = os.path.join(base_dir, "annotations_augmented.json")
    with open(output_path, 'w') as f:
        json.dump(new_coco_data, f, indent=4)

    print(f"\nAugmentation complete! A new file with a total of {len(new_coco_data['images'])} image entries has been created:")
    print(output_path)
    print(f"The augmented images have been saved to the '{augmented_image_dir}' folder.")

if __name__ == "__main__":
    train_dir = ""
    augment_dataset(train_dir)