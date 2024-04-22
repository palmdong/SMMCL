import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from PIL import Image 

from config import config
from engine.logger import get_logger
from engine.evaluator import Evaluator
from dataloader.RGBXDataset import RGBXDataset
from dataloader.dataloader import ValPre
from models.builder import EncoderDecoder as segmodel
from utils.pyt_utils import ensure_dir, parse_devices  
from utils.visualize import print_iou    
from utils.metric import hist_info, compute_score


logger = get_logger()


class SegEvaluator(Evaluator):
    def func_per_iteration(self, data, device):
        img = data['data']
        label = data['label']
        modal_x = data['modal_x']
        name = data['fn']

        pred = self.sliding_eval_rgbX(img, modal_x, config.eval_crop_size, config.eval_stride_rate, device) 
        hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes, pred, label) 
        results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp, 'correct': correct_tmp}

        if self.save_path is not None:
            ensure_dir(self.save_path+'_color')

            fn = name + '.png'

            # Save Color Results
            result_img = Image.fromarray(pred.astype(np.uint8), mode='P')  
            class_colors = self.dataset.get_class_colors()
            palette_list = list(np.array(class_colors).flat)
            if len(palette_list) < 768:
                palette_list += [0] * (768 - len(palette_list))
            result_img.putpalette(palette_list)
            result_img.save(os.path.join(self.save_path+'_color', fn))

        return results_dict

    def compute_metric(self, results): 
        hist = np.zeros((config.num_classes, config.num_classes))
        correct = 0
        labeled = 0
        count = 0
        for d in results:
            hist += d['hist']
            correct += d['correct']
            labeled += d['labeled']
            count += 1

        iou, mean_IoU, mean_IoU_wo_background, freq_IoU, mean_pixel_acc, pixel_acc = compute_score(hist, correct, labeled)  
        result_line = print_iou(iou, freq_IoU, mean_pixel_acc, pixel_acc,
                                dataset.class_names, show_wo_background=True)  # True for LLRGBD. False for NYU and MFNet.
        return result_line

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-d', '--devices', default='0', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('-p', '--save_path', default=None)


    args = parser.parse_args()
    all_dev = parse_devices(args.devices)

    network = segmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d) 


    data_setting = {'rgb_root': config.rgb_root_folder,
                    'rgb_format': config.rgb_format,
                    'gt_root': config.gt_root_folder,
                    'gt_format': config.gt_format,
                    'transform_gt': config.gt_transform,  
                    'x_root':config.x_root_folder,
                    'x_format': config.x_format,
                    'x_single_channel': config.x_is_single_channel,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source,
                    'class_names': config.class_names}
    val_preprocess = ValPre()
    dataset = RGBXDataset(data_setting, 'val', val_preprocess) 
 
    with torch.no_grad():
        segmentor = SegEvaluator(dataset, config.num_classes, config.norm_mean,
                                 config.norm_std, network,
                                 config.eval_scale_array, config.eval_flip,
                                 all_dev, args.verbose, args.save_path)
        segmentor.run(config.checkpoint_dir, args.epochs, config.val_log_file,
                      config.link_val_log_file)  
