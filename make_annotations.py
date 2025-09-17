import json
import os
from pathlib import Path

def create_augmented_json_with_sam_paths(original_sam_json_path, augmented_json_path):
    """
    Copies the original SAM-based JSON file and changes only the image paths
    to create a JSON file for the augmented dataset.
    """
    try:
        with open(original_sam_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Original file not found - {original_sam_json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: JSON file format is incorrect - {original_sam_json_path}")
        return

    # Iterate through the 'images' list and change the 'file_name' path
    for image_info in data.get("images", []):
        if "file_name" in image_info and isinstance(image_info["file_name"], str):
            original_file_path = Path(image_info["file_name"])
            original_filename_stem = original_file_path.stem
            
            # Remove the 'synthetic_' prefix from the filename
            if original_filename_stem.startswith("synthetic_"):
                original_filename_stem = original_filename_stem.replace("synthetic_", "", 1)
            
            # Create a new filename and path
            new_filename = f"synthetic_{original_filename_stem}_aug.jpg"
            image_info["file_name"] = f"images_augmented/{new_filename}"

    try:
        with open(augmented_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print(f"✅ JSON file paths successfully updated: {augmented_json_path}")
    except Exception as e:
        print(f"Error: An issue occurred while saving the file: {e}")

if __name__ == "__main__":
    # --- Path settings ---
    original_sam_json = ""
    output_augmented_json = ""
    
    # Execute the function
    create_augmented_json_with_sam_paths(original_sam_json, output_augmented_json)