import torch 
from torch import nn
import math

from torchvision import models
from collections import OrderedDict
from .dinov2 import vit_base


class DinoVITBackbone(nn.Module):
    def __init__(self, backbone_type, pretrained=True):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dino:main', backbone_type, pretrained=pretrained)
        self.embed_dim = self.backbone.embed_dim

    def forward(self, x):
        x = self.backbone.prepare_tokens(x)
        for blk in self.backbone.blocks:
            x = blk(x)
        x = self.backbone.norm(x)
        x = x[:, 1:]
        x = x.permute(0,2,1) 
        b, c, side = x.shape
        x = x.view(b, c, int(math.sqrt(side)), int(math.sqrt(side)))
        return x

    
class DinoV2Backbone(nn.Module):
    def __init__(self, backbone_type, pretrained=True):
        super().__init__()
        self.backbone = vit_base(
                            patch_size=16,
                            init_values=1.0,
                            block_chunks=4
                        ) 
        self.embed_dim = self.backbone.embed_dim

    def forward(self, x):
        x = self.backbone.forward_features(x)['x_norm_patchtokens']
        x = x.permute(0,2,1) 
        b, c, side = x.shape
        x = x.view(b, c, int(math.sqrt(side)), int(math.sqrt(side)))
        return x 


class VITBackbone(nn.Module):
    def __init__(self, backbone_type, pretrained=True):
        super().__init__()
        self.backbone = models.vit_b_16(weights='DEFAULT')
        self.backbone.heads.head = nn.Identity() 
        self.embed_dim = self.backbone.hidden_dim

    def forward(self, x):
        # Reshape and permute the input tensor
        n = x.shape[0]
        x = self.backbone._process_input(x)
        batch_class_token = self.backbone.class_token.expand(n, -1, -1)

        # Expand the class token to the full batch
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.backbone.encoder(x)
        x = x[:, 1:]
        x = x.permute(0,2,1)
        b, c, side = x.shape
        x = x.view(b, c, int(math.sqrt(side)), int(math.sqrt(side)))
        return x


class SWINBackbone(nn.Module):
    def __init__(self, backbone_type, pretrained=True):
        super().__init__()
        if backbone_type == 'swint':
            backbone = models.swin_v2_t(weights='DEFAULT')
        if backbone_type == 'swins':
            backbone = models.swin_v2_s(weights='DEFAULT') 
        self.embed_dim = 768
        self.backbone = nn.Sequential(OrderedDict([*(list(backbone.named_children())[:-3])]))

    def forward(self, x):
        return self.backbone(x)


class ResNetBackbone(nn.Module):
    def __init__(self, backbone_type='resnet50', pretrained=True):
        super().__init__()
        if backbone_type == 'resnet18':
            backbone = models.resnet18(weights='DEFAULT')
            self.embed_dim = 512
        elif backbone_type == 'resnet50':
            backbone = models.resnet50(weights='DEFAULT')
            self.embed_dim = 2048
        elif backbone_type == 'resnet101':
            backbone = models.resnet101(weights='DEFAULT')
            self.embed_dim = 2048
        # take output before average pooling layer
        self.backbone = nn.Sequential(OrderedDict([*(list(backbone.named_children())[:-2])]))

    def forward(self, x):
        return self.backbone(x) 

BACKBONE_MAP = {'resnet50': lambda: ResNetBackbone(backbone_type='resnet50', pretrained=True),
                'resnet101': lambda: ResNetBackbone(backbone_type='resnet101', pretrained=True),
                'swins': lambda: SWINBackbone(backbone_type='swins', pretrained=True),
                'swint': lambda: SWINBackbone(backbone_type='swint', pretrained=True),
                'vitb': lambda: VITBackbone(backbone_type='vitb16'),
                'dino_vitb16': lambda: DinoVITBackbone(backbone_type='dino_vitb16', pretrained=True),
                'dinov2_vitb16': lambda: DinoV2Backbone(backbone_type='dinov2_vitb16'),
              }

class BackboneBase(nn.Module):
    def __init__(self, backbone_type='resnet50', pretrained=True, finetune=False):
        super().__init__()
        if backbone_type in BACKBONE_MAP:
            self.backbone = BACKBONE_MAP[backbone_type]()
        else:
            raise ValueError(f"Backbone type '{backbone_type}' is not implemented. Supported types: {list(BACKBONE_MAP.keys())}")
        
        self.out_dim = getattr(self.backbone, 'embed_dim', 768)
        # Full train or finetune backbone parameters
        if finetune:
            # Freeze earlier blocks
            for block in self.backbone.backbone.blocks[:-1]:  # Adjust -2 to control how many layers to fine-tune
                for param in block.parameters():
                    param.requires_grad = False

            # Ensure the last block remain trainable
            for block in self.backbone.backbone.blocks[-1:]:  
                for param in block.parameters():
                    param.requires_grad = True


    def forward(self, x):
        return self.backbone(x)