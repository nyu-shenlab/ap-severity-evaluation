import os
import math
import random

import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader, default_collate
from .torch_transforms import compose_transform
from .dataloader_2D import LazyLoadDatasetCSV2D


def validation_collate_fn(batch):
    """
    Make sure the validation batch_size = 1
    Returns:
    tensor: Batch of image tensors
    """
    assert len(batch)==1
    images = []
    #phases = []

    # assume batch size for validation is always 1
    item = batch[0]
    for i in range(len(item[0])):
        images.append(torch.stack([item[0][i]]))
        #phases.append(torch.stack([item[2][i]]))
    
    rid = torch.Tensor([item[2]])
    
    if isinstance(batch[0][1],torch.Tensor):
        labels_batch = torch.stack([item[1] for item in batch])
    else:
        labels_batch = torch.tensor([item[1] for item in batch])
    
    return images, labels_batch, rid


def load_data(
    datalist_path: str,
    num_workers: int,
    image_resize: int = 256,
    vis_ids: list = None,
    val_batch_size: int = 1,
):
    """
    Load test and optional visualization datasets

    Args:
        datalist_path (str): Path to CSV containing image data.
        num_workers (int): Number of workers for DataLoader.
        image_resize (int): Resize image to (image_resize, image_resize).
        vis_ids (list, optional): List of RIDs to visualize.
        val_batch_size (int): Batch size for visualization.

    Returns:
        Tuple[DataLoader, Optional[DataLoader]]
    """
    print(vis_ids)
    test_data = pd.read_csv(datalist_path)
    test_data = test_data[test_data['rid'].isin(vis_ids)]
    vis_data = test_data[test_data['rid'].isin(vis_ids)] if vis_ids else None

    test_dataset = LazyLoadDatasetCSV2D(
        test_data,
        transform=compose_transform(None, (image_resize, image_resize))
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=validation_collate_fn
    )
    print('Test loader:', len(test_dataloader))

    if vis_data is not None:
        visualization_dataset = LazyLoadDatasetCSV2D(
            vis_data,
            transform=compose_transform(None, (image_resize, image_resize))
        )
        visualization_dataloader = DataLoader(
            visualization_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=False,
            collate_fn=validation_collate_fn
        )
        print('Visualization loader:', len(visualization_dataloader))
    else:
        visualization_dataloader = None

    return test_dataloader, visualization_dataloader


