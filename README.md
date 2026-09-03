# SMMCL
PyTorch implementation of "Understanding Dark Scenes by Contrasting Multi-Modal Observations"  
[[WACV paper](https://openaccess.thecvf.com/content/WACV2024/papers/Dong_Understanding_Dark_Scenes_by_Contrasting_Multi-Modal_Observations_WACV_2024_paper.pdf), [supp](https://drive.google.com/file/d/11DVmw8t92OT3lTF53RiUNAZfw_Yp3OTi/view?usp=drive_link)]


## Updates
**[2024/08/02]** The supplementary material has been updated. Previously there were errors in Figures III, V, and VI.
**[2024/04/22]** The source code for the low-light indoor setting (the LLRGBD dataset) has been released.  
**[2023/10/24]** Our paper has been accepted to WACV 2024. See you in Hawaii, USA.  

## Preparation
- Python 3.10.6, Torch 1.12.1, CUDA 10.2, requirements.txt  
- Download the [datasets](https://drive.google.com/drive/folders/19p2zjc0UPnKZh06D5tNsILLB1aJvjf1l?usp=sharing) and put them in /datasets.  

## Train
Download the pretrained weight of [SegNext-B](https://cloud.tsinghua.edu.cn/d/c15b25a6745946618462/) and put it in /pretrained/segnext.  
```bash
cd /path/to/SMMCL_LLRGBD or /path/to/SMMCL_MFNet or /path/to/SMMCL_NYU
# modify config.py
python train.py -d 0-3 
```

## Test 
Quick Start: Download our pretrained [weights](https://drive.google.com/drive/folders/1wSX5vLr78_rfDV6-lCqAEFScMWqAjKkv?usp=sharing) and put them in /SMMCL_XXX/log_XXX_mscan_b/checkpoint.
```bash
cd /path/to/SMMCL_LLRGBD
python eval.py -d 0-3 -e 500 
```
```bash
cd /path/to/SMMCL_MFNet
python eval.py -d 0-3 -e 300 
```
```bash
cd /path/to/SMMCL_NYU
python eval.py -d 0-3 -e 600
```

## Acknowledgement
Our code was built based on the repositories of [CMX](https://github.com/huaaaliu/RGBX_Semantic_Segmentation) and [MSCSCL](https://github.com/RViMLab/ECCV2022-multi-scale-and-cross-scale-contrastive-segmentation/tree/main). We thank the authors for their efforts.

## Citation
```
@InProceedings{SMMCL_2024_WACV,
  author    = {Dong, Xiaoyu and Yokoya, Naoto},
  title     = {Understanding Dark Scenes by Contrasting Multi-Modal Observations},
  booktitle = {WACV},
  year      = {2024}
}
```
