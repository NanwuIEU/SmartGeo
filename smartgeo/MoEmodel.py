import torch
import timm
import numpy as np
import torch.nn as nn
from smartgeo import helper
import torch.nn.functional as F
import math
# from torchsummary import summary

class VPRModel(nn.Module):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.

    Args:
        pl (_type_): _description_
    """

    def __init__(self,
                 # ---- Backbone 
                 model_name='dinov2_vitb14_MoEMixVPR',
                 backbone_arch='dinov2_vitb14',
                 pretrained=True,
                 layers_to_freeze=1,
                 layers_to_crop=[],
                 layer1=20,
                 use_cls=False,
                 norm_descs=True,

                 # ---- Aggregator 
                 agg_arch='MoEMixVPR',  # CosPlace, NetVLAD, GeM
                 agg_config={},
                 ):
        super().__init__()
        self.pretrained = pretrained  
        self.layers_to_freeze = layers_to_freeze  
        self.layers_to_crop = layers_to_crop  # layers_to_crop=[4],  # 4 crops the last resnet layer, 3 crops the 3rd, ...etc
        self.layer1 = layer1
        self.use_cls = use_cls
        self.norm_descs = norm_descs
        self.agg_config = agg_config 
        # self.save_hyperparameters()  # write hyperparams into a file
        self.model_name = model_name

        self.batch_acc = []  # we will keep track of the % of trivial pairs/triplets at the loss level

        # ----------------------------------
        # get the backbone and the aggregator 
        self.backbone = helper.get_backbone(backbone_arch, pretrained, layer1=self.layer1, use_cls=self.use_cls,
                                            norm_descs=self.norm_descs)
        self.aggregator = helper.get_aggregator(agg_arch, agg_config)
        # self.DDI = DDI(input_shape=(1024, 768), dropout=0.2, patch=16, alpha=0.1, layernorm=True)

    # the forward pass of the lightning model
    def forward(self, x):
        x = self.backbone(x)
        # x = self.DDI(x)
        x = self.aggregator(x)
        return x



class TimmModel(nn.Module):

    def __init__(self,
                 model_name='dinov2_vitb14_MixVPR',
                 pretrained_path=None,
                 backbone_arch='',
                 pretrained=True,
                 img_size=224,
                 layer1=8,
                 # Aggregator 
                 agg_arch='MoEMixVPR',
                 agg_config={},
                 ):

        super(TimmModel, self).__init__()

        self.img_size = img_size

        if "dino" in backbone_arch:
            self.model = VPRModel(backbone_arch=backbone_arch, agg_arch=agg_arch, layer1=layer1, agg_config=agg_config)
        elif "vitt" in backbone_arch:
            # automatically change interpolate pos-encoding to img_size
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size)
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if pretrained_path:
          
            state_dict = torch.load(pretrained_path)
            print("Start from:", pretrained_path)
            self.load_state_dict(state_dict)
        

    # def get_config(self):
    #     # data_config = timm.data.resolve_model_data_config(self.model)
    #     # data_config = self.model.default_cfg
    #     data_config = {'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]}
    #     return data_config
    
    
    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)
    
    def get_config(self,):
        data_config = timm.data.resolve_model_data_config(self.model)
        return data_config

        
    def forward(self, img1, img2=None):
        
        if img1 is not None and img2 is not None:
       
            image_features1 = self.model(img1)     
            image_features2 = self.model(img2)
            
            return image_features1, image_features2            
              
        else:
            image_features = self.model(img1)
            return image_features
