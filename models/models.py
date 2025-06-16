import math, copy
import torch
from torch import Tensor
from torch import nn
from torchvision import models

from .backbones.backbones import BackboneBase
        
class AttentionPoolingLayer(nn.Module):
    def __init__(self, in_features, out_classes=1):
        super(AttentionPoolingLayer, self).__init__()
        out_features = in_features//4
        self.fc1 = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Tanh())
        self.fc2 = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Sigmoid())
        self.fc3 = nn.Linear(out_features, out_classes)
            

    def forward(self, x):
        # print(x.shape)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        # print(x1.shape, x2.shape)
        att_logit = self.fc3(x1 * x2)
        # print(att_logit.shape)
        attn_wts = nn.Softmax(dim=1)(att_logit)
    
        x = torch.sum(attn_wts.unsqueeze(-1) @ x.unsqueeze(-2), dim=1)
        return x, attn_wts
        

class MLP(nn.Module):
    def __init__(self, input_dim=86, output_dim=2, hidden_dim=512, dropout_prob=0.2):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)


class CustomModelWS(nn.Module):
    def __init__(self, 
            backbone_type='resnet50', 
            pretrained=False, 
            finetune=False, 
            n_out=2, 
            dropout_prob=0.1, 
            top_k = 0.1
            ):
        super(CustomModelWS, self).__init__()
        self.backbone_type = backbone_type
        self.backbone = BackboneBase(backbone_type, pretrained, finetune)
        self.dropout = nn.Dropout(dropout_prob)
        self.in_features = self.backbone.out_dim  # Typically 2048 for resnet
        self.top_k = top_k

        self.conv1x1_layer = nn.Conv2d(in_channels=self.in_features, out_channels=1, kernel_size=1)
        self.agg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.pooling_layer = AttentionPoolingLayer(self.in_features, n_out)

        # Define classification mapping based on label mode
        self.map_classification = nn.Identity()

    def top_t_percent_pooling(self, input_tensor, t=0.25):
        flattened_tensor = input_tensor.view(input_tensor.size(0), input_tensor.size(1), -1)
        num_elements_to_keep = max(int(flattened_tensor.size(-1) * t), 1)  # Ensure at least one element is kept
        sorted_tensor, _ = torch.sort(flattened_tensor, dim=-1, descending=True)
        top_values = sorted_tensor[:, :, :num_elements_to_keep]
        pooled_value = torch.mean(top_values, dim=-1, keepdim=True)
        output_tensor = pooled_value.view(input_tensor.size(0), input_tensor.size(1), 1, 1)
        return output_tensor


    def extract_image_feature(self, x):
        # Process image features in a large batch size
        x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])  # (B*K) x 3 x H x W
 
        # Determine sub-batch size based on backbone type
        if 'swin' in self.backbone_type or 'vitb' in self.backbone_type:
            sub_batch_size = 96
        elif 'sam' in self.backbone_type: 
            sub_batch_size = 96
        else:
            sub_batch_size = 16
        
        num_chunks = (x.shape[0] + sub_batch_size - 1) // sub_batch_size  # Calculate number of chunks
        # Use torch.chunk to divide into specified number of chunks
        sub_batches = torch.chunk(x, num_chunks)
        sub_batch_outputs = []
        i=0
        for sub_batch in sub_batches:
            i+=1
            #print(i,sub_batch.shape)
            # Pass each sub-batch through the backbone
            sub_batch_output = self.backbone(sub_batch)
            sub_batch_output = self.dropout(sub_batch_output)
            sub_batch_outputs.append(sub_batch_output)

        x = torch.cat(sub_batch_outputs, dim=0)  # (B*K) x in_features x H x W
        return x

    def forward(self, x):
        batch_size, K  = x.shape[0], x.shape[1]
        
        # Go through backbone in batches
        x = self.extract_image_feature(x) 
        BK, in_features, H, W = x.shape

        # Apply the 1x1 convolution to get the saliency map
        sal_map = self.conv1x1_layer(x)  # (B*K) x 1 x H x W

        # Aggregate image features
        x = self.agg(x)  # (B*K) x (in_features + 768) x 1 x 1
        x = x.squeeze(-1).squeeze(-1)  # (B*K) x (in_features + 768)
        x = x.view(batch_size, K, -1)  # B x K x (in_features + 768)

        # Apply pooling layer
        pooled_output, attn_wts = self.pooling_layer(x)  # pooled_output shape depends on pooling layer

        agg_sal_map = self.top_t_percent_pooling(sal_map, t=self.top_k)  # Shape: (B*K) x 1 x 1 x 1 
        agg_sal_map = agg_sal_map.view(batch_size, -1, agg_sal_map.shape[1])  # Adjust shape as needed

        # Reshape sal_map for matrix multiplication
        sal_map = sal_map.view(batch_size, -1, sal_map.shape[1], sal_map.shape[2], sal_map.shape[3])

        # Compute the final output
        op = (attn_wts.transpose(1, 2) @ agg_sal_map).squeeze(-1)

        op = self.map_classification(op)
        return op, (attn_wts, agg_sal_map, sal_map)
