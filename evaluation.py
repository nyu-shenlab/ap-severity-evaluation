"""Acute pancreatitis severity inference on an arbitrary CT datalist.

Reproduces the evaluation path of ``ap-severity/bin/train.py --phase evaluation
--select_all_phases`` (no test-time augmentation) and the model ensemble of the
final NYU analysis: every patient in the datalist is scored by every model on all
of its axial phases, and the per-phase logits are combined exactly as in
training-time validation. Labels are optional, so the same script runs on any
cohort that follows the datalist schema in ``dataloader/dataloader.py``.
"""
import os, json
import argparse
import logging
import numpy as np
import pandas as pd
import yaml

import torch
from torch import nn

import pytorch_lightning as pl
from dataloader import dataloader as dl
# Not named ``utils``: the DINO torch.hub repo does ``from utils import ...`` and
# would pick up this package instead of its own.
from eval_utils.logging import setup_logger
from eval_utils.utils import *
from eval_utils.visualization_utils import *
from pytorch_grad_cam import GradCAM
from models.lightning_models import Custom_Model_Lightning


def initialize_model(config_path, ckpt_path, logger: logging.Logger):
    r"""initialize_model(config_path, ckpt_path, logger) -> Custom_Model_Lightning

    Build a model from its training ``config.yaml`` and load its checkpoint.

    Args:
        config_path (str): ``config.yaml`` saved next to the checkpoint at training time.
        ckpt_path (str): Lightning checkpoint (e.g. ``best_val_auc_S.ckpt``).
        logger (logging.Logger): Logger.

    Returns:
        Custom_Model_Lightning: Model in eval mode, with its config on ``.config``.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file) or {}
    # This code only implements the 2D ordinal attention-pooling model; anything
    # else would load but silently produce meaningless probabilities.
    for key, expected in [('model_type', '2D'), ('label_mode', 'ordinal'), ('pool_type', 'attention')]:
        if config.get(key, expected) != expected:
            raise ValueError(f"{config_path}: {key}={config[key]!r}, only {expected!r} is supported")
    if config.get('clinical_fusion_type'):
        raise ValueError(f"{config_path}: clinical fusion models are not supported")

    # Inference without trained weights would quietly return random scores.
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = Custom_Model_Lightning(config=config)
    logger.info(f"Loading checkpoint at {ckpt_path}")
    weights = torch.load(ckpt_path, weights_only=False, map_location='cpu')['state_dict']
    model.load_state_dict(weights)
    model.config = config
    return model.eval()


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
    cam = GradCAM(model=wrapped_model, target_layers=[target_layer])

    for jdx, batch in enumerate(visualization_dataloader):
        batch = ToDevice(batch, device)
        images_list, label, rid = batch[0], batch[1], int(batch[2].item())
        label = torch.sum(label, dim=-1).item()

        # Process each phase
        for phase_idx, phase_images in enumerate(images_list):
            phase_images = phase_images.squeeze(0).float()
            logger.info(f"Phase {phase_idx} input shape: {phase_images.shape}")

            # GradCAM needs gradients even though the model is in eval mode.
            with torch.enable_grad():
                grayscale_cam = cam(input_tensor=phase_images, targets=None)

            for img_idx in range(len(phase_images)):
                original_image = phase_images[img_idx].cpu().numpy()  # [C, H, W]
                cam_image = grayscale_cam[img_idx]  # [H, W]
                original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())

                create_visualizations(gradcam_dir,
                            original_image,
                            cam_image,
                            rid,
                            phase_idx,
                            img_idx,
                            label)
        logger.info(f"GradCAM visualizations done for patient {jdx + 1}")

    logger.info(f"GradCAM visualizations saved to {gradcam_dir}")


def run_evaluation(models, test_dataloader, args, logger: logging.Logger):
    r"""run_evaluation(models, test_dataloader, args, logger) -> Dict[str, dict]

    Score every patient with every model in a single pass over the data.

    Args:
        models (Dict[str, Custom_Model_Lightning]): Models keyed by name.
        test_dataloader (DataLoader): One patient (all phases) per batch.
        args (argparse.Namespace): Parsed arguments.
        logger (logging.Logger): Logger.

    Returns:
        Dict[str, dict]: Per model, ``rids``, ``labels`` (ordinal pairs, ``[-1, -1]``
        if unknown) and ``predictions`` rows ``[P(mild), P(moderate), P(severe)]``.
    """
    logger.info(f"Starting evaluation of {len(models)} model(s) on {len(test_dataloader)} patients")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for model in models.values():
        model.to(device)

    rids, labels = [], []
    preds = {name: [] for name in models}
    for jdx, batch in enumerate(test_dataloader):
        batch = ToDevice(batch, device)
        with torch.no_grad():
            for name, model in models.items():
                preds[name].append(model.validation_step(batch, jdx).cpu().numpy())
        rids.append(int(batch[2].item()))
        labels.append(batch[1].cpu().numpy())
        if (jdx + 1) % 50 == 0:
            logger.info(f"Scored {jdx + 1}/{len(test_dataloader)} patients")

    labels = np.concatenate(labels, axis=0).tolist()
    results = {}
    for name in models:
        results[name] = {'rids': rids,
                         'predictions': np.concatenate(preds[name], axis=0).tolist(),
                         'labels': labels}
        path = os.path.join(args.output_path, f"{args.prediction_name}_{name}.json")
        with open(path, "w") as f:
            json.dump(results[name], f, indent=2)
        logger.info(f"Saved {name} predictions to {path}")
    return results


def minmax(x):
    r"""minmax(x) -> np.ndarray

    Rescale to [0, 1] over the cohort, matching ``sklearn.MinMaxScaler`` (a constant
    column maps to 0).
    """
    span = x.max() - x.min()
    return (x - x.min()) / span if span > 0 else np.zeros_like(x)


def ensemble(results, model_specs, args, logger: logging.Logger):
    r"""ensemble(results, model_specs, args, logger) -> pd.DataFrame

    Combine per-model predictions the way the final NYU analysis did.

    Severe score: mean over the severe models of P(severe), each first min-max
    scaled over the scored cohort. Mild score: plain mean over the mild models of
    P(mild). Because of the min-max step, a patient's severe score depends on the
    rest of the cohort; it is a ranking within the cohort, not a probability.

    Args:
        results (Dict[str, dict]): Output of :func:`run_evaluation`.
        model_specs (Dict[str, List[dict]]): ``severe`` / ``mild`` model entries.
        args (argparse.Namespace): Parsed arguments.
        logger (logging.Logger): Logger.

    Returns:
        pd.DataFrame: One row per patient: ``rid``, ``mild_pred``, ``severe_pred``
        and, when labels are known, ``mild_label`` / ``severe_label``.
    """
    sev = [np.array(results[m['name']]['predictions'])[:, 2] for m in model_specs['severe']]
    mild = [np.array(results[m['name']]['predictions'])[:, 0] for m in model_specs['mild']]
    if len(sev[0]) < 10:
        logger.warning(f"Only {len(sev[0])} patients: min-max scaled severe scores are unstable on tiny cohorts")

    any_name = model_specs['severe'][0]['name']
    out = pd.DataFrame({'rid': results[any_name]['rids'],
                        'mild_pred': np.mean(mild, axis=0),
                        'severe_pred': np.mean([minmax(s) for s in sev], axis=0)})
    label_sum = np.array(results[any_name]['labels']).sum(axis=1)
    known = label_sum >= 0
    out['mild_label'] = np.where(known, (label_sum == 0).astype(float), np.nan)
    out['severe_label'] = np.where(known, (label_sum == 2).astype(float), np.nan)
    assert out.rid.is_unique, 'a patient was scored more than once'

    path = os.path.join(args.output_path, f"{args.prediction_name}_ensemble.csv")
    out.to_csv(path, index=False)
    logger.info(f"Saved ensemble predictions for {len(out)} patients to {path}")
    return out


def load_model_specs(args):
    r"""load_model_specs(args) -> Dict[str, List[dict]]

    Resolve the models to run, either from ``--models_file`` (ensemble) or from
    ``--config_path`` / ``--pretrain_weights`` (single model).
    """
    if args.models_file is None:
        return {'single': [{'name': 'model', 'config': args.config_path, 'weights': args.pretrain_weights}]}

    with open(args.models_file, 'r') as f:
        spec = yaml.safe_load(f)
    root = args.weights_root or spec.get('weights_root', '')
    specs = {}
    for task in ['severe', 'mild']:
        specs[task] = []
        for i, entry in enumerate(spec[task]):
            model_dir = os.path.join(root, entry['model_dir'])
            specs[task].append({'name': f"{task}{i}",
                                'config': os.path.join(model_dir, 'config.yaml'),
                                'weights': os.path.join(model_dir, entry['weight'])})
    return specs


def main(args):
    os.makedirs(args.output_path, exist_ok=True)
    log_path = os.path.join(args.output_path, f'{args.prediction_name}_{args.phase}.log')
    logger = setup_logger(log_path)
    logger.info(f"Logs will be saved at: {log_path}")

    # Inference without TTA is deterministic; the seed only pins GradCAM / any stray randomness.
    pl.seed_everything(args.random_seed, workers=True)

    model_specs = load_model_specs(args)
    models = {m['name']: initialize_model(m['config'], m['weights'], logger)
              for task in model_specs.values() for m in task}
    logger.info(f"Torch CUDA Device Count: {torch.cuda.device_count()}")

    # All released models were trained at one resolution; refuse to mix.
    image_sizes = {m.config['image_size'] for m in models.values()}
    if len(image_sizes) != 1:
        raise ValueError(f"Models disagree on image_size: {image_sizes}")

    test_dl, vis_dl = dl.load_data(
        datalist_path = args.datalist_path,
        num_workers = args.num_workers,
        image_resize = image_sizes.pop(),
        vis_ids = args.visualization_ids,
        val_batch_size = 1,
    )
    logger.info("DataLoaders prepared successfully")

    if args.phase == 'evaluate':
        results = run_evaluation(models, test_dl, args, logger)
        if args.models_file is not None:
            ensemble(results, model_specs, args, logger)
    elif args.phase == 'visualization':
        if len(models) != 1 or vis_dl is None:
            raise ValueError("visualization needs a single model and --visualization_ids")
        run_visualization(next(iter(models.values())), vis_dl, args, logger)
    else:
        raise ValueError(f"Unknown phase: {args.phase}")


if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--datalist_path", type=str, required=True, help="slice-level CSV, see dataloader/dataloader.py")
    parser.add_argument("--output_path", type=str, default="output")
    parser.add_argument("--phase", type=str, default="evaluate", choices=["evaluate", "visualization"])
    parser.add_argument("--prediction_name", type=str, default='predictions')

    # Either an ensemble ...
    parser.add_argument("--models_file", type=str, default=None, help="YAML listing severe/mild models, e.g. configs/ensemble.yaml")
    parser.add_argument("--weights_root", type=str, default=None, help="overrides weights_root in --models_file")
    # ... or a single model
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--pretrain_weights", type=str, default=None)

    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--random_seed", type=int, default=47)
    parser.add_argument("--visualization_ids", nargs='+', type=int, default=None, help="List of visualization IDs (e.g., --visualization_ids 20037 70023 20062)")

    args = parser.parse_args()
    if args.models_file is None and (args.config_path is None or args.pretrain_weights is None):
        parser.error("give --models_file, or both --config_path and --pretrain_weights")
    main(args)
