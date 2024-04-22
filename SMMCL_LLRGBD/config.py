import os
import os.path as osp
import numpy as np
import sys
import time
import argparse
from easydict import EasyDict as edict


C = edict()
config = C      

C.seed = 12345  

remoteip = os.popen('pwd').read()
C.root_dir = osp.abspath(osp.join(os.getcwd(), './'))  


###### Dataset ######   
# # Get Images
C.dataset_name = 'LLRGBD' 
C.dataset_path = '/path/to/datasets/LLRGBD/train'    
# C.dataset_path = '/path/to/datasets/LLRGBD/test'   
C.rgb_root_folder = osp.join(C.dataset_path, 'rgb')  
C.rgb_format = '.png'
C.x_root_folder = osp.join(C.dataset_path, 'depth')
C.x_format = '.png'
C.x_is_single_channel = True
C.gt_root_folder = osp.join(C.dataset_path, 'label') 
C.gt_format ='.png'
C.gt_transform = False  # False for LLRGBD and MFNet, True for NYU. 
C.train_source = osp.join(C.dataset_path, "train.txt")
C.eval_source = osp.join(C.dataset_path, "test.txt") 
C.is_test = False
C.num_train_imgs = 1418  
C.num_eval_imgs = 479
C.num_classes = 14  
C.class_names =  ['void', 
                  'bed','books','ceiling','chair','floor','furniture', 
                  'objects','painting','sofa','table','tv','wall', 
                  'window']

# Ignore for Training 
C.background = 255  

# Crop for Training
C.image_height = 480  
C.image_width = 640  

# Normalization
C.norm_mean = np.array([0.485, 0.456, 0.406])  
C.norm_std = np.array([0.229, 0.224, 0.225])  


###### Model ######   
C.pretrained_model = '/path/to/pretrained/segnext/mscan_b.pth'
C.backbone = 'mscan_b'     
C.decoder = 'LightHamHead'  
C.decoder_embed_dim = 512  


###### Training ######  
# SMMCL 
C.min_views_per_class = 5   
C.max_views_per_class = 300  
C.max_features_total = 4800  
C.cm_temperature = 0.1
C.temperature = 0.1   
C.weight_cm = 0.2
C.weight_vis = 0.2
C.weight_aux = 0.2

# Settings
C.nepochs = 1000  
C.batch_size = 16    
C.niters_per_epoch = C.num_train_imgs // C.batch_size  + 1 
C.num_workers = 16 
C.optimizer = 'AdamW'
C.lr = 6e-5         
C.weight_decay = 0.01
C.momentum = 0.9     
C.lr_power = 0.9     
C.warm_up_epoch = 10 
C.fix_bias = True     
C.bn_eps = 1e-3      
C.bn_momentum = 0.1
C.train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75]  

# Save Checkpoint 
C.checkpoint_start_epoch = 300
C.checkpoint_step = 100



###### Evaluation ######  
C.eval_crop_size = [480, 640] 
C.eval_stride_rate = 2 / 3   
C.eval_scale_array = [1]  
C.eval_flip = False       


###### Log and Checkpoint ######
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path(osp.join(C.root_dir))  

C.log_dir = osp.abspath('log_' + C.dataset_name + '_' + C.backbone)  
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.abspath(osp.join(C.log_dir, "checkpoint"))

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_file + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'

if __name__ == '__main__':
    print(config.nepochs)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-tb', '--tensorboard', default=False, action='store_true')
    args = parser.parse_args()

    if args.tensorboard:
        open_tensorboard()
