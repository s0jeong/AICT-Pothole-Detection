import os
import cv2
import glob
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

def predict_test():
    # # --- Addition: Register test set metadata for visualization ---
    # try:
    #     register_coco_instances("pothole_test", {}, "path/to/dummy_annotations.json", "/home/dromii2/pothole/dataset/test/images")
    # except AssertionError:
    #     print("Dataset 'pothole_test' is already registered. Skipping.")
    # MetadataCatalog.get("pothole_test").set(thing_classes=["pothole"])
    # # ---------------------------------------------------

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    # <--- Change: Comment out the line that forces CPU to allow GPU usage ---
    # cfg.MODEL.DEVICE = "cpu"
    
    cfg.MODEL.WEIGHTS = "model_final.pth"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    
    cfg.INPUT.MIN_SIZE_TEST = 1024
    cfg.INPUT.MAX_SIZE_TEST = 1600
    
    predictor = DefaultPredictor(cfg)
    
    pothole_metadata = MetadataCatalog.get("pothole_test")
    
    # Predict test images
    test_image_dir = ""
    output_dir = ""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving results to the '{output_dir}' folder...")
    for img_path in glob.glob(os.path.join(test_image_dir, "*.jpg")) + glob.glob(os.path.join(test_image_dir, "*.JPG")):
        img = cv2.imread(img_path)
        outputs = predictor(img)
        
        v = Visualizer(img[:, :, ::-1], metadata=pothole_metadata, scale=0.8)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        
        output_path = os.path.join(output_dir, os.path.basename(img_path))
        cv2.imwrite(output_path, out.get_image()[:, :, ::-1])
        print(f"Saved prediction: {output_path}")

if __name__ == "__main__":
    predict_test()