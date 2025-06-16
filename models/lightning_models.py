import torch
from torch import Tensor
from torch import nn

import pytorch_lightning as pl
from collections import OrderedDict

from torchmetrics.classification import MulticlassAUROC, MulticlassF1Score, MulticlassAveragePrecision, MulticlassAccuracy, BinaryF1Score, BinaryAUROC, BinaryAccuracy, BinaryAveragePrecision
from torchmetrics import MetricCollection

from .models import CustomModelWS

class Custom_Model_Lightning(pl.LightningModule):
    def __init__(self, config = {}):
        super(Custom_Model_Lightning, self).__init__()
        self.save_hyperparameters()

        self.model = CustomModelWS(
            backbone_type = config["backbone"],
            pretrained = True,
            finetune = False,
            n_out = 2,
            top_k = config["top_k"],
            dropout_prob = config["dropout_prob"],)

    def forward(self, x_bag):
        return self.model(x_bag)
    
        
    def prepare_for_metrics(self, y_hat, y):
      
        y_hat = torch.sigmoid(y_hat)
        preds_M = 1 - y_hat[:,0]
        preds_MS = (1 - y_hat[:,1]) * y_hat[:,0]
        preds_S = y_hat[:,1] * y_hat[:,0] 
        preds = torch.stack([preds_M,preds_MS,preds_S],dim=-1)
        labels = torch.sum(y, dim=-1).long()

        return preds, labels
    

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1].float()
     
        phases_y_hat = []
        for i in range(len(x)):
            model_out = self.forward(x[i].float())
            y_hat = model_out[0].squeeze(1)
            phases_y_hat.append(y_hat)
          
        k = min(1,len(phases_y_hat))
        y_hat = torch.mean(torch.topk(torch.stack(phases_y_hat),k=k,dim=0)[0],dim=0)


        # Update validation metrics
        preds, labels = self.prepare_for_metrics(y_hat, y)

        return preds

