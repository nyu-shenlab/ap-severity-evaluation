import time, os, json
import argparse
import logging 
import numpy as np
from collections import OrderedDict

import torch
from torch import nn
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

import pytorch_lightning as pl
from dataloader import dataloader as dl 
from utils.logging import setup_logger
from utils.utils import *
from utils.visualization_utils import *
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from models.lightning_models import Custom_Model_Lightning

def initialize_model(args, config, logger: logging.Logger):
    """Initialize the Lightning model (with or without pretrained weights)."""

    # Seed everything for reproducibility
    torch.cuda.manual_seed(args.random_seed)
    pl.seed_everything(args.random_seed, workers=True)
    torch.autograd.set_detect_anomaly(True)

    # Create your model
    model = Custom_Model_Lightning(config = config)

    # Check if there's a saved checkpoint
    ckpt_path = args.pretrain_weights
    if os.path.exists(ckpt_path):
        logger.info(f"Loading checkpoint at {ckpt_path}")
        weights = torch.load(ckpt_path, weights_only=False)['state_dict']
        model.load_state_dict(weights)
    else:
        logger.info("No existing checkpoint found, starting from scratch.")

    return model


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        # Ensure input is float32
        x = x.float()
        # Add batch dimension if not present
        if len(x.shape) == 3:  # [C, H, W]
            x = x.unsqueeze(0)  # [1, C, H, W]
        elif len(x.shape) == 4:  # [K, C, H, W]
            x = x.unsqueeze(0)  # [1, K, C, H, W]
        return self.model(x)[0]  # Return only predictions, not the tuple


def run_visualization(model, visualization_dataloader, args, logger: logging.Logger):
    logger.info("Starting visualization")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    # Create output directory for GradCAM visualizations
    gradcam_dir = os.path.join(args.output_path, 'gradcam_visualizations')
    os.makedirs(gradcam_dir, exist_ok=True)

    # Initialize GradCAM with wrapped model
    wrapped_model = ModelWrapper(model.model)
    target_layer = model.model.conv1x1_layer
    cam = GradCAM(model=wrapped_model, target_layers=[target_layer])#, use_cuda=torch.cuda.is_available())

    for jdx, batch in enumerate(visualization_dataloader):
        # Assume batch is (images, labels) or (images, labels, possibly-other-data)
        batch = ToDevice(batch, device)
        images_list, label, rid = batch[0], batch[1].item(), int(batch[2].item())

        # Process each phase
        for phase_idx, phase_images in enumerate(images_list):
            # Ensure input is float32
            phase_images = phase_images.squeeze(0).float()
            logger.info(f"Phase {phase_idx} input shape: {phase_images.shape}")

            # Get model predictions and GradCAM for this phase
            # Enable gradients temporarily for GradCAM
            with torch.enable_grad():
                grayscale_cam = cam(input_tensor=phase_images, targets=None)
            
            # Convert to numpy and save visualizations
            for img_idx in range(len(phase_images)):
                # Get original image and GradCAM
                original_image = phase_images[img_idx].cpu().numpy()  # [C, H, W]
                cam_image = grayscale_cam[img_idx]  # [H, W]
                
                # Normalize original image for visualization
                original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())
                
                create_visualizations(gradcam_dir,
                            original_image, 
                            cam_image,
                            rid,
                            phase_idx,
                            img_idx,
                            label)
        logger.info(f"GradCAM visualizations for {rid}")

    logger.info(f"GradCAM visualizations saved to {gradcam_dir}")


def run_evaluation(model, test_dataloader, args, logger: logging.Logger):
    """Evaluate or run inference on the test set."""
    logger.info("Starting evaluation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    preds = []
    labels = []

    for jdx, batch in enumerate(test_dataloader):
        # Assume batch is (images, labels) or (images, labels, possibly-other-data)
        batch = ToDevice(batch, device)
        images_list, label = batch[0], batch[1]

        # Get model predictions
        with torch.no_grad():
            pred_y = model.validation_step(batch, jdx)
        
        pred_y_np = pred_y.cpu().detach().numpy()
        label_np = label.cpu().detach().numpy()

        preds.append(pred_y_np)
        labels.append(label_np)
        
    preds_arr = np.concatenate(preds, axis=0)
    labels_arr = np.concatenate(labels, axis=0)

    # ---------- FINAL AVERAGE ACROSS TTA EPOCHS ----------
    final_results = {
        'predictions': preds_arr.tolist(),
        'labels': labels_arr.tolist()
    }

    final_path = os.path.join(args.output_path, f"{args.prediction_name}.json")
    with open(final_path, "w") as f:
        json.dump(final_results, f, indent=2)  # pretty-print with 2 spaces
    logger.info(f"Evaluation complete. Averaged predictions saved to {final_path}.")
    


def main(args):
    # 1. Set up logger as soon as possible
    os.makedirs(args.output_path, exist_ok=True)
    #model_name = os.path.split(args.pretrain_weights)[0]
    #args.output_path = os.path.join(args.output_path,model_name)

    log_path = f'{args.phase}_log.log'
    logger = setup_logger(log_path)
    logger.info(f"Logs will be saved at: {log_path}")

    # 2. Update hyperparameters if available

    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file) or {}  # Handle empty YAML
    logger.info(f"Load config from {args.config_path}")

    # 4. Initialize model
    model = initialize_model(args, config, logger)
    device_count = torch.cuda.device_count()
    logger.info(f"Torch CUDA Device Count: {device_count}")

    # 5. Prepare dataloaders
    test_dl, vis_dl = dl.load_data(
        datalist_path = args.datalist_path,
        num_workers = args.num_workers,
        image_resize = 256,
        vis_ids = args.visualization_ids,
        val_batch_size = 1,
    )
    logger.info("DataLoaders prepared successfully")

    # 6. Training or Evaluation mode
    if args.phase == 'evaluate':
        run_evaluation(model, test_dl, args, logger)
    elif args.phase == 'visualization':
        run_visualization(model, vis_dl, args, logger)
    else:
        logger.error(f"Unknown phase: {args.phase}")


if __name__=='__main__':

    parser = argparse.ArgumentParser()

    # Arguments
    parser.add_argument("--datalist_path", type=str, default="/gpfs/data/shenlab/Yanqi/ap-severity-project/data/df_final.csv")
    parser.add_argument("--output_path", type=str, default="/gpfs/data/shenlab/Yanqi/ap-severity-project/data/df_final.csv")
    parser.add_argument("--config_path", type=str, default="/gpfs/data/shenlab/Yanqi/ap-severity-project/save/test")
    parser.add_argument("--pretrain_weights", type=str, default=None)
    parser.add_argument("--phase", type=str, default="evaluation") 

    # Data arguments
    parser.add_argument("--image_size", type=int, default=256)

    parser.add_argument("--val_batchsize", type=int, default=1, help="validation batch size")
    parser.add_argument("--num_workers", type=int, default=8) 
    parser.add_argument("--random_seed", type=int, default=47)
    parser.add_argument("--prediction_name", type=str, default='predictions')
    parser.add_argument("--visualization_ids", nargs='+', type=int, default=None, help="List of visualization IDs (e.g., --visualization_ids 20037 70023 20062)")
    
    args = parser.parse_args()
    main(args)
