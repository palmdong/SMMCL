# SMMCL
PyTorch implementation of "Understanding Dark Scenes by Contrasting Multi-Modal Observations"  
[WACV paper, supp] [[arXiv](https://arxiv.org/abs/2308.12320)]


## Updates
**[2023/08/23]** Paper is available on [arXiv](https://arxiv.org/abs/2308.12320).   
**[2023/10/24]** Papar was accepted by WACV2024. We thank anonymous reviewers from ICCV2023 and WACV2024 for their suggestions. See you in Hawaii! 


## Abstract
Understanding dark scenes based on multi-modal image data is challenging, as both the visible and auxiliary modalities provide limited semantic information for the task. Previous methods focus on fusing the two modalities but neglect the correlations among semantic classes when minimizing losses to align pixels with labels, resulting in inaccurate class predictions. To address these issues, we introduce a supervised multi-modal contrastive learning approach to increase the semantic discriminability of the learned multi-modal feature spaces by jointly performing cross-modal and intra-modal contrast under the supervision of the class correlations. The cross-modal contrast encourages same-class embeddings from across the two modalities to be closer and pushes different-class ones apart. The intra-modal contrast forces same-class or different-class embeddings within each modality to be together or apart. We validate our approach on a variety of tasks that cover diverse light conditions and image modalities. Experiments show that our approach can effectively enhance dark scene understanding based on multi-modal images with limited semantics by shaping semantic-discriminative feature spaces. Comparisons with previous methods demonstrate our state-of-the-art performance. 


<p align="center"> <img src="figs/figi_result_low.png" width="75%"> </p>

<p align="center"> <img src="figs/figii_result_night.png" width="74.5%"> </p>

<p align="center"> <img src="figs/figiii_result_normal.png" width="88%"> </p>

## Citation
```
@InProceedings{Dong2023SMMCL,
  author    = {Dong, Xiaoyu and Yokoya, Naoto},
  title     = {Understanding Dark Scenes by Contrasting Multi-Modal Observations},
  booktitle = {arXiv preprint arXiv:2308.12320},
  year      = {2023}
}
```
