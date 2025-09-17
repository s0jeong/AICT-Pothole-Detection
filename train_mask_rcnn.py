import os
import wandb
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.events import get_event_storage

# --- Custom Hook for Wandb Logging ---
class WandbHook(HookBase):
    def after_step(self):
        storage = get_event_storage()
        log_data = {}
        for k, (v, _) in storage.latest_with_smoothing_hint(20).items():
            log_data[k] = v
        
        if "bbox/AP" in storage.latest():
            for k, (v, _) in storage.latest().items():
                if "AP" in k:
                    log_data[k] = v

        wandb.log(log_data)

# Register datasets
def register_datasets():
    dataset_info = {
        "pothole_train_sam": {
            "image_dir": "train",
            "anno_file": "train/annotations_sam.json"
        },
        "pothole_train_augmented": {
            "image_dir": "train",
            "anno_file": "train/annotations_augmented.json"
        },
        "pothole_val": {
            "image_dir": "val/images",
            "anno_file": "val/annotations_sam.json"
        }
    }
    for name, info in dataset_info.items():
        try:
            register_coco_instances(name, {}, info["anno_file"], info["image_dir"])
        except AssertionError:
            print(f"Dataset '{name}' is already registered. Skipping.")


# Inherit from DefaultTrainer to add a Hook
class WandbTrainer(DefaultTrainer):
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, WandbHook())
        return hooks

# Training setup
def train_model():
    register_datasets()
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    # --- Wandb project settings ---
    wandb.init(
        project="pothole-detection",
        name="r50_70epoch_combined",
        config=cfg
    )
    # --------------------------------

    cfg.DATASETS.TRAIN = ("pothole_train_sam", "pothole_train_augmented")
    cfg.DATASETS.TEST = ("pothole_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.001

    total_images = 1456
    epochs = 70
    max_iter = int(epochs * total_images / cfg.SOLVER.IMS_PER_BATCH)
    cfg.SOLVER.MAX_ITER = max_iter
    
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.SOLVER.CHECKPOINT_PERIOD = 5000
    cfg.INPUT.MIN_SIZE_TRAIN = (1024, 1333)
    cfg.INPUT.MAX_SIZE_TRAIN = 1600
    cfg.INPUT.MIN_SIZE_TEST = 1024
    cfg.INPUT.MAX_SIZE_TEST = 1600
    cfg.MODEL.RPN.ANCHOR_SIZES = [[8, 16, 32, 64, 128]]
    cfg.MODEL.RPN.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]

    cfg.OUTPUT_DIR = ""
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = WandbTrainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()

    wandb.finish()

if __name__ == "__main__":
    train_model()