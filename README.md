# SMMCL
PyTorch implementation of "Understanding Dark Scenes by Contrasting Multi-Modal Observations"  
[paper, supp] [arXiv]


## Updates
**[2023/08/23]** Paper will be available on arXiv soon. For any questions or discussions, reach Xiaoyu Dong at dong@ms.k.u-tokyo.ac.jp.


## Abstract
Understanding dark scenes based on multi-modal image data is challenging, as both the visible and auxiliary modalities provide limited semantic information for the task. Previous methods focus on fusing the two modalities but neglect the correlations among semantic classes when minimizing losses to align pixels with labels, resulting in inaccurate class predictions. To address these issues, we introduce a supervised multi-modal contrastive learning approach to increase the semantic discriminability of the learned multi-modal feature spaces by jointly performing cross-modal and intra-modal contrast under the supervision of the class correlations. The cross-modal contrast encourages same-class embeddings from across the two modalities to be closer and pushes different-class ones apart. The intra-modal contrast forces same-class or different-class embeddings within each modality to be together or apart. We validate our approach on a variety of tasks that cover diverse light conditions and image modalities. Experiments show that our approach can effectively enhance dark scene understanding based on multi-modal images with limited semantics by shaping semantic-discriminative feature spaces. Comparisons with previous methods demonstrate our state-of-the-art performance. 


## Citation
```
@InProceedings{Dong2023SMMCL,
  author    = {Dong, Xiaoyu and Yokoya, Naoto},
  title     = {Understanding Dark Scenes by Contrasting Multi-Modal Observations},
  booktitle = {},
  year      = {2023}
}
```
