import os
import cv2
import numpy as np
import torch
import torch.utils.data as data
from pickletools import uint8   

class RGBXDataset(data.Dataset):  
    def __init__(self, setting, split_name, preprocess=None, file_length=None):
        super(RGBXDataset, self).__init__()
        self._rgb_path = setting['rgb_root']
        self._rgb_format = setting['rgb_format']
        self._gt_path = setting['gt_root']
        self._gt_format = setting['gt_format']
        self._transform_gt = setting['transform_gt']
        self._x_path = setting['x_root']
        self._x_format = setting['x_format']
        self._x_single_channel = setting['x_single_channel']
        self._train_source = setting['train_source']
        self._eval_source = setting['eval_source']
        self.class_names = setting['class_names']

        self._split_name = split_name
        self.preprocess = preprocess
        self._file_length = file_length 
        
        self._file_names = self._get_file_names(split_name)

    def __len__(self):   
        if self._file_length is not None:
            return self._file_length
        return len(self._file_names)

    def __getitem__(self, index):       
        if self._file_length is not None:
            item_name = self._construct_new_file_names(self._file_length)[index]
        else:
            item_name = self._file_names[index]

        # item_name = item_name.split('.')[0][4:]  # Modify item_name for NYU. Comment for LLRGBD and MFNet.

        rgb_path = os.path.join(self._rgb_path, item_name + self._rgb_format)
        x_path = os.path.join(self._x_path, item_name + self._x_format)
        gt_path = os.path.join(self._gt_path, item_name + self._gt_format)

        # Read Images
        rgb = self._open_image(rgb_path, cv2.COLOR_BGR2RGB)  

        gt = self._open_image(gt_path, cv2.IMREAD_GRAYSCALE, dtype=np.uint8)  
        if self._transform_gt:
            gt = self._gt_transform(gt)   

        if self._x_single_channel:
            x = self._open_image(x_path, cv2.IMREAD_GRAYSCALE)
            x = cv2.merge([x, x, x])  
        else:
            x = self._open_image(x_path, cv2.COLOR_BGR2RGB)
        
        # Preprocess for Training/Val
        if self.preprocess is not None:
            rgb, gt, x = self.preprocess(rgb, gt, x)

        if self._split_name == 'train':
            rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float()
            gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
            x = torch.from_numpy(np.ascontiguousarray(x)).float()

        output_dict = dict(data=rgb, label=gt, modal_x=x, fn=str(item_name), n=len(self._file_names))

        return output_dict

    def _get_file_names(self, split_name): 
        assert split_name in ['train', 'val']
        source = self._train_source
        if split_name == "val":
            source = self._eval_source

        file_names = []
        with open(source) as f:
            files = f.readlines()

        for item in files:
            file_name = item.strip()
            file_names.append(file_name)

        return file_names

    def _construct_new_file_names(self, length):
        assert isinstance(length, int)
        files_len = len(self._file_names)                          
        new_file_names = self._file_names * (length // files_len)   

        rand_indices = torch.randperm(files_len).tolist()
        new_indices = rand_indices[:length % files_len]

        new_file_names += [self._file_names[i] for i in new_indices]

        return new_file_names

    def get_length(self):  
        return self.__len__()

    @staticmethod
    def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None):
        img = np.array(cv2.imread(filepath, mode), dtype=dtype)
        return img

    @staticmethod
    def _gt_transform(gt):  
        return gt - 1 

    @classmethod
    def get_class_colors(*args):  
        void      = [0, 0, 0] # black
        bed       = [0, 0, 1] # vivid blue
        books     = [0.9137,0.3490,0.1882] # orange
        ceiling   = [0, 0.8549, 0] # green
        chair     = [0.5843,0,0.9412] # purple
        floor     = [0.8706,0.9451,0.0941] # yellow
        furniture = [1.0000,0.8078,0.8078] # light pink
        objects   = [0,0.8784,0.8980] # vivid blue
        painting  = [0.4157,0.5333,0.8000] # light blue purple
        sofa      = [0.4588,0.1137,0.1608] # brown red  
        table     = [0.9412,0.1373,0.9216] # vivid pink
        tv        = [0,0.6549,0.6118] # lake green
        wall      = [0.9765,0.5451,0] # carrot
        window    = [0.8824,0.8980,0.7608] # rice yellow

        class_colors = np.array([void,bed,books,ceiling,chair,floor,furniture,objects,painting,sofa,table,tv,wall,window])
        class_colors = np.uint8(class_colors*255)
        return class_colors