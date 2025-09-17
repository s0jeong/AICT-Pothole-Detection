import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import OrderedDict
import wandb

# Detectron2 libraries
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances

def visualize_evaluation_results(results: OrderedDict, save_path: str = "evaluation_results.png"):
    """
    Visualizes the results of the Detectron2 COCOEvaluator as a heatmap and saves the image file.
    """
    # Convert Bbox and Segm results to a DataFrame
    bbox_metrics = {k: v for k, v in results['bbox'].items() if isinstance(v, float)}
    segm_metrics = {k: v for k, v in results['segm'].items() if isinstance(v, float)}
    
    df_bbox = pd.DataFrame.from_dict(bbox_metrics, orient='index', columns=['Bbox'])
    df_segm = pd.DataFrame.from_dict(segm_metrics, orient='index', columns=['Segm'])
    
    df = pd.concat([df_bbox, df_segm], axis=1)

    # Create the heatmap
    plt.figure(figsize=(12, 5))
    sns.heatmap(df.T, annot=True, fmt=".1f", cmap="viridis", linewidths=.5, cbar=True)
    
    # Set the graph title and labels
    plt.title('COCO Evaluation Metrics Heatmap (mAP)', fontsize=16, fontweight='bold')
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(rotation=0, fontsize=12)

    # Adjust layout and save the file
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n✅ Evaluation results heatmap saved to '{save_path}'.")
    plt.close() # Close the window to prevent it from popping up.

def main():
    """
    Main function for Detectron2 model evaluation and result visualization.
    """
    # --- 1. Configuration (User can modify this section) ---
    # Dataset name and paths
    dataset_name = "pothole_val"
    val_annotations_path = "val/annotations_sam.json"
    val_images_path = "val/images"
    
    # Path to the model to be evaluated
    model_weights_path = "model_final.pth"
    
    # Model configuration file
    config_file = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    
    # Output path for results (automatically set based on model path)
    output_dir = os.path.join(os.path.dirname(model_weights_path), "evaluation")
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 2. Initialize Wandb ---
    wandb.init(
        project="pothole-detection",
        name=f"evaluation_{os.path.basename(os.path.dirname(model_weights_path))}",
        job_type="evaluation"
    )
    
    # --- 3. Register the dataset ---
    try:
        register_coco_instances(dataset_name, {}, val_annotations_path, val_images_path)
        print(f"Dataset '{dataset_name}' registered.")
    except AssertionError:
        print(f"Dataset '{dataset_name}' is already registered. Skipping.")

    # --- 4. Load model configuration (Config) ---
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.DATASETS.TEST = (dataset_name,)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_weights_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    
    # --- 5. Prepare the model and evaluator ---
    print("\nLoading model and preparing evaluator...")
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator(dataset_name, output_dir=output_dir)
    val_loader = build_detection_test_loader(cfg, dataset_name)

    # --- 6. Run evaluation ---
    print("Starting evaluation...")
    results = inference_on_dataset(predictor.model, val_loader, evaluator)
    print("\n=== Final Evaluation Results (COCO mAP) ===")
    print(results)
    
    # --- 7. Visualize results and log to Wandb ---
    if results:
        # Log numerical results to Wandb
        wandb.log(results)
        
        # Generate and save the chart image
        chart_save_path = os.path.join(output_dir, "evaluation_metrics_chart.png")
        visualize_evaluation_results(results, save_path=chart_save_path)
        
        # Log the chart image to Wandb
        wandb.log({"Evaluation Metrics Heatmap": wandb.Image(chart_save_path)})

    # --- 8. Finish Wandb ---
    wandb.finish()

if __name__ == "__main__":
    main()