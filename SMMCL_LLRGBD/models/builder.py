import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.init_func import init_weight    
from engine.logger import get_logger

from .smmcl_loss import SupervisedMultiModalContrastiveLoss as smmcl_loss 


logger = get_logger()


class EncoderDecoder(nn.Module):
    def __init__(self, cfg=None, criterion=None, norm_layer=None):  
        super(EncoderDecoder, self).__init__()                    

        self.channels = [64, 128, 320, 512]  
        self.norm_layer = norm_layer 

        # Backbone 
        if cfg.backbone == 'mscan_l':                         
            logger.info('Using Backbone: SegNeXt-L')
            from .encoders.dual_segnext import mscan_l as backbone
            self.backbone = backbone(init_cfg=None)       
        elif cfg.backbone == 'mscan_b':                      
            logger.info('Using Backbone: SegNeXt-B')
            from .encoders.dual_segnext import mscan_b as backbone
            self.backbone = backbone(init_cfg=None)  
        else:                                              
            logger.info('Using Backbone: SegNeXt-B')       
            from .encoders.dual_segnext import mscan_b as backbone
            self.backbone = backbone(init_cfg=None)

        # Decoder
        if cfg.decoder == 'LightHamHead':         
            logger.info('Using Decoder: LightHamHead')
            from .decoders.LightHamHead import LightHamHead
            self.channels = [128, 320, 512]        
            self.decode_head = LightHamHead(in_channels=self.channels, in_index=[1, 2, 3], channels=cfg.decoder_embed_dim, ham_channels=cfg.decoder_embed_dim, dropout_ratio=0.1, num_classes=cfg.num_classes) 
        else:
            logger.info('No Decoder Specified. Using MLPDecoder.') 
            from .decoders.MLPDecoder import DecoderHead
            self.decode_head = DecoderHead(in_channels=self.channels, num_classes=cfg.num_classes, norm_layer=norm_layer, embed_dim=cfg.decoder_embed_dim)

        # Loss
        self.criterion = criterion
        if self.criterion:         
            self.init_weights(cfg, pretrained=cfg.pretrained_model) 
            
            self.smmcl_loss = smmcl_loss(cfg) 
            self.weight_cm, self.weight_vis, self.weight_aux = cfg.weight_cm, cfg.weight_vis, cfg.weight_aux
        
    # Weight Initilization for Pretrained Model
    def init_weights(self, cfg, pretrained=None):
        # Backbone
        if pretrained:
            logger.info('Loading Pretrained Model: {}'.format(pretrained))  
            self.backbone.init_weights(pretrained=pretrained) 
        logger.info('Initializing Weights......')
        # Decoder
        init_weight(self.decode_head, nn.init.kaiming_normal_,  
                self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                mode='fan_in', nonlinearity='relu')  

    # Define Segmentation Function
    def encode_decode(self, rgb, modal_x):
        """Encode images with backbone and decode into a semantic map  
        map of the same size as input."""
        orisize = rgb.shape
        x, color_fea, other_fea = self.backbone(rgb, modal_x)
        out = self.decode_head.forward(x)
        out = F.interpolate(out, size=orisize[2:], mode='bilinear', align_corners=False) 
        return out, color_fea, other_fea

    def forward(self, rgb, modal_x, label=None):
        out, color_fea, other_fea = self.encode_decode(rgb, modal_x)
        if label is not None:                      
            loss = self.criterion(out, label.long())
            loss_cm, loss_vis, loss_aux = self.smmcl_loss(label, color_fea, other_fea) 
            return loss + loss_cm*self.weight_cm + loss_vis*self.weight_vis + loss_aux*self.weight_aux
        return out  