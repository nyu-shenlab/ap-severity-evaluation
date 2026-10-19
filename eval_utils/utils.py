import os
import json
import logging
from collections import OrderedDict
import argparse
import yaml

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    return {
        "Total parameters": total_params,
        "Trainable parameters": trainable_params,
        "Non-trainable parameters": non_trainable_params,
    }

def ToDevice(batch, device):
    batch = list(batch)
    batch[0] = [item.to(device) for item in batch[0]]
    batch[1] = batch[1].to(device)
    batch[2] = batch[2].to(device) if batch[2] is not None else batch[2]
    return batch