# SMMCL
PyTorch implementation of "Understanding Dark Scenes by Contrasting Multi-Modal Observations"  
[[WACV paper](https://openaccess.thecvf.com/content/WACV2024/papers/Dong_Understanding_Dark_Scenes_by_Contrasting_Multi-Modal_Observations_WACV_2024_paper.pdf), [supp](https://drive.google.com/file/d/11DVmw8t92OT3lTF53RiUNAZfw_Yp3OTi/view?usp=drive_link)]


## Updates
**[2023/10/24]** Paper was accepted by WACV2024. We thank anonymous reviewers from ICCV2023 and WACV2024 for their suggestions to our paper. See you in Hawaii.   
**[2024/04/22]** The code for experiments on low-light indoor scenes (the LLRGBD dataset) was uploaded. 


## Abstract
Understanding dark scenes based on multi-modal image data is challenging, as both the visible and auxiliary modalities provide limited semantic information for the task. Previous methods focus on fusing the two modalities but neglect the correlations among semantic classes when minimizing losses to align pixels with labels, resulting in inaccurate class predictions. To address these issues, we introduce a supervised multi-modal contrastive learning approach to increase the semantic discriminability of the learned multi-modal feature spaces by jointly performing cross-modal and intra-modal contrast under the supervision of the class correlations. The cross-modal contrast encourages same-class embeddings from across the two modalities to be closer and pushes different-class ones apart. The intra-modal contrast forces same-class or different-class embeddings within each modality to be together or apart. We validate our approach on a variety of tasks that cover diverse light conditions and image modalities. Experiments show that our approach can effectively enhance dark scene understanding based on multi-modal images with limited semantics by shaping semantic-discriminative feature spaces. Comparisons with previous methods demonstrate our state-of-the-art performance. 


<p align="center"> <img src="figs/figi_result_low.png" width="75%"> </p>

<p align="center"> <img src="figs/figii_result_night.png" width="74.5%"> </p>

<p align="center"> <img src="figs/figiii_result_normal.png" width="88%"> </p>

## Experiments

### Preparation
- Python 3.10.6, Torch 1.12.1, CUDA 10.2, requirements.txt  
- Download the [datasets](https://drive.google.com/drive/folders/19p2zjc0UPnKZh06D5tNsILLB1aJvjf1l?usp=sharing) and put them in /datasets.  

### Train
Download the pretrained weight of [SegNext-B](https://cloud.tsinghua.edu.cn/d/c15b25a6745946618462/) and put it in /pretrained/segnext.  
```bash
cd /path/to/SMMCL_LLRGBD or /path/to/SMMCL_MFNet or /path/to/SMMCL_NYU
# modify config.py
python train.py -d 0-3 
```

### Test 
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

### Acknowledgement
Our code was built based on the repositories of [CMX](https://github.com/huaaaliu/RGBX_Semantic_Segmentation) and [MSCSCL](https://github.com/RViMLab/ECCV2022-multi-scale-and-cross-scale-contrastive-segmentation/tree/main). We thank the authors for their efforts.

## Citation
```
@InProceedings{Dong2024SMMCL,
  author    = {Dong, Xiaoyu and Yokoya, Naoto},
  title     = {Understanding Dark Scenes by Contrasting Multi-Modal Observations},
  booktitle = {WACV},
  year      = {2024}
}
```
