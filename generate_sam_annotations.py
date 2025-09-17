import os
import json
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
from pycocotools import mask as mask_util
from tqdm import tqdm

def generate_sam_annotations(
    base_dir,
    json_file="annotations.json",
    output_json_file="annotations_sam.json",
    model_type="vit_h",
    sam_checkpoint="sam_vit_h_4b8939.pth"
):
    """
    Converts COCO-format Bbox annotations to Polygon annotations using SAM.
    """
    print("Loading SAM model... This may take some time.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    print("SAM model loaded successfully.")

    # Load original COCO annotation file
    annotations_path = os.path.join(base_dir, json_file)
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)

    new_annotations = []
    
    # Show progress using tqdm
    for ann in tqdm(coco_data['annotations'], desc="Processing annotations"):
        image_id = ann['image_id']
        image_info = next((item for item in coco_data['images'] if item['id'] == image_id), None)
        if not image_info:
            continue
            
        image_path = os.path.join(base_dir, "images", image_info['file_name'])
        if not os.path.exists(image_path):
            print(f"Warning: Image file not found: {image_path}")
            continue

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        predictor.set_image(image)
        
        # Convert Bbox to SAM's input format
        bbox = ann['bbox']
        input_box = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
        
        # Predict mask with SAM
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,  # Get only the most probable mask
        )
        
        # Create a new annotation
        new_ann = ann.copy()
        binary_mask = masks[0]
        
        # Convert mask to polygon
        contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        segmentation = []
        for contour in contours:
            # Convert to [x1, y1, x2, y2, ...] list format for COCO
            if contour.size >= 6:  # Check if it's a valid polygon
                segmentation.append(contour.flatten().tolist())
        
        if not segmentation:
            # If no valid polygon is found, fall back to the original bbox (safety measure)
            x, y, w, h = ann['bbox']
            new_ann['segmentation'] = [[x, y, x, y+h, x+w, y+h, x+w, y]]
        else:
            new_ann['segmentation'] = segmentation

        # Recalculate area and bbox from RLE format (more accurate)
        rle = mask_util.encode(np.asfortranarray(binary_mask))
        new_ann['area'] = float(mask_util.area(rle))
        new_ann['bbox'] = list(mask_util.toBbox(rle))
            
        new_annotations.append(new_ann)

    # Replace existing annotation data with the new data
    coco_data['annotations'] = new_annotations

    # Save to a new file
    output_path = os.path.join(base_dir, output_json_file)
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=4)
        
    print(f"\nSuccessfully created new annotation file: {output_path}")


if __name__ == "__main__":
    # Convert train dataset
    print("Starting train dataset conversion...")
    train_dir = ""
    generate_sam_annotations(train_dir)
    
    # Convert val dataset
    print("\nStarting validation dataset conversion...")
    val_dir = ""
    generate_sam_annotations(val_dir)