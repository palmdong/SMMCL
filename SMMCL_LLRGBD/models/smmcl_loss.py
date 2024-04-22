import torch
import torch.nn as nn

from engine.logger import get_logger


logger = get_logger()  


def has_inf_or_nan(x):
    return torch.isinf(x).max().item(), torch.isnan(x).max().item()


class SupervisedMultiModalContrastiveLoss(nn.Module): 
    def __init__(self, config):
        super().__init__() 
        
        self.num_all_classes = config.num_classes  
        self.min_views_per_class = config.min_views_per_class  
        self.max_views_per_class = config.max_views_per_class  
        self.max_features_total = config.max_features_total    
        
        self.cross_modal_temperature = config.cm_temperature 
        self.temperature = config.temperature  

        self._scale = None 


    def forward(self, label:torch.Tensor, color_fea:torch.Tensor, other_fea:torch.Tensor):  
        
        with torch.no_grad():  
            scale = int(label.shape[-1] // color_fea.shape[-1])  
            class_distribution, dominant_classes = self.get_dist_and_classes(label, scale)   
            
        color_feas, labels = self.sample_anchors_fast(dominant_classes, color_fea) 
        other_feas, _ = self.sample_anchors_fast(dominant_classes, other_fea)   # Note that, labels for color_feas and other_feas are the same.
        
        loss_cm = self.cm_contrastive_loss(color_feas, other_feas, labels)
        loss_vis = self.contrastive_loss(color_feas, labels) 
        loss_aux = self.contrastive_loss(other_feas, labels)   
        return loss_cm, loss_vis, loss_aux    
    
    def get_dist_and_classes(self, label:torch.Tensor, scale:int) -> torch.Tensor: 
        """Determines the distribution of the classes in each scale*scale patch of the ground truth label N-H-W,
        for given experiment, returning class_distribution as N-C-H//scale-W//scale tensor. 

        Also determines dominant classes in each patch of the ground truth label N-H-W, based on the class distribution. 
        Output is N-C-H//scale-W//scale where C might be 1 (just one dominant class) or more.

        If label_scaling_mode == 'nn' peforms nearest neighbour interpolation on the label without one_hot encoding and
        returns N-1-H//scale-W//scale
        """
        n, h, w = label.shape   
        self._scale = scale   
        label_down = torch.nn.functional.interpolate(label.unsqueeze(1).float(), (h//scale, w//scale), mode='nearest') 
        return label_down.long(), label_down.long() 

    def sample_anchors_fast(self, dominant_classes, feature):  
        """
        self.anchors_per_image =  

        input:  dominant_classes N-1-H-W           
                features  N-C-H-W

        return: sampled_features T-C-V 
                sampled_labels   T  
                T: classes/anchors in BATCH (with repetition)
                C: feature space dimensionality
                V: views_per_class/anchor, i.e. samples from each class/anchor
        """
        n = dominant_classes.shape[0]  
        c = feature.shape[1]  
        feature = feature.view(n, c, -1) 
        dominant_classes = dominant_classes.view(n, -1)  
        cls_in_batch = []    
        cls_counts_in_batch = []    

        classes_ids = torch.arange(start=0, end=self.num_all_classes, step=1, device=dominant_classes.device)
        compare = dominant_classes.unsqueeze(-1) == classes_ids.unsqueeze(0).unsqueeze(0)
        cls_counts = compare.sum(1)

        present_ids = torch.where(cls_counts[:, :] >= self.min_views_per_class) 
        batch_ids, cls_in_batch = present_ids  

        min_views = torch.min(cls_counts[present_ids]) 
        total_cls = cls_in_batch.shape[0] 

        cls_counts_in_batch = cls_counts  
        views_per_class = self._select_views_per_class(min_views, total_cls, cls_in_batch, cls_counts_in_batch)                 
        sampled_features = torch.zeros((total_cls, c, views_per_class), dtype=torch.float).cuda() 
        sampled_labels = torch.zeros(total_cls, dtype=torch.float).cuda() 

        for i in range(total_cls):  
            indices_from_cl = compare[batch_ids[i], :, cls_in_batch[i]].nonzero().squeeze()         
            random_permutation = torch.randperm(indices_from_cl.shape[0]).cuda() 
            sampled_indices_from_cl = indices_from_cl[random_permutation[:views_per_class]] 
            sampled_features[i] = feature[batch_ids[i], :, sampled_indices_from_cl]
            sampled_labels[i] = cls_in_batch[i]
        return sampled_features, sampled_labels     
    
    def _select_views_per_class(self, min_views, total_cls, cls_in_batch, cls_counts_in_batch): 
        if self.max_views_per_class == 1:  
            # no capping to views_per_class
            views_per_class = min_views
        else: 
            # capping views_per_class to avoid OOM
            views_per_class = min(min_views, self.max_views_per_class)
            if views_per_class == self.max_views_per_class:  
                logger.info(                                 
                    f'\n capping views_per_class to {self.max_views_per_class},'  
                    f' cls_and_counts: {cls_in_batch} {cls_counts_in_batch} ')

        if views_per_class * total_cls > self.max_features_total:
            views_per_class = self.max_features_total // total_cls
            logger.info(  
                f'\n'  
                f' capping total features  to {self.max_features_total} total_cls:  {total_cls} '
                f'--> views_per_class:  {views_per_class} ,'
                f'  cls_and_counts: {cls_in_batch} {cls_counts_in_batch}')
        return views_per_class    

    def contrastive_loss(self, feats, labels):   
        """
        input:  feats T-C-V
                      T: classes/anchors in BATCH (with repetition)
                      C: feature space dimensionality
                      V: views_per_class/anchor, i.e. samples from each class/anchor
                labels T
        return: loss
        """
        # prepare feats 
        feats = torch.nn.functional.normalize(feats, p=2, dim=1)  
        feats = feats.transpose(dim0=1, dim1=2)  
        num_anchors, views_per_anchor, c = feats.shape  

        labels = labels.contiguous().view(-1, 1)  
        labels_ = labels.repeat(1, views_per_anchor) 
        labels_ = labels_.view(-1, 1) 

        pos_mask, neg_mask = self.get_masks(labels_, num_anchors, views_per_anchor)  
        feats_flat = feats.contiguous().view(-1, c) 
        dot_product = torch.div(torch.matmul(feats_flat, torch.transpose(feats_flat, 0, 1)), self.temperature)
        loss = self.InfoNCE_loss(pos_mask, neg_mask, dot_product)  
        return loss
    
    
    @staticmethod
    def get_masks(labels, num_anchors, views_per_anchor):  
        """
        takes flattened labels and identifies pos/neg of each anchor
        :param labels: T*V-1
        :param num_anchors: T
        :param views_per_anchor: V
        :return: pos_mask, neg_mask
        """
        # extract mask indicating same-class samples
        mask = torch.eq(labels, torch.transpose(labels, 0, 1)).float()  
        neg_mask = 1 - mask  # indicator of negatives
        # set diagonal mask elements to zero -- self-similarities
        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(num_anchors * views_per_anchor).view(-1, 1).cuda(),
                                                     0)
        pos_mask = mask * logits_mask  # indicator of positives
        return pos_mask, neg_mask
    
    
    def InfoNCE_loss(self, pos, neg, dot):
        """
        :param pos: V*T-V*T
        :param neg: V*T-V*T
        :param dot: V*T-V*T
        :return:
        """
        # logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = dot  # - logits_max.detach()

        neg_logits = torch.exp(logits) * neg
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)
        # print('exp_logits ', has_inf_or_nan(exp_logits))
        log_prob = logits - torch.log(exp_logits + neg_logits)
        # print('log_prob ', has_inf_or_nan(log_prob))
        pos_sums = pos.sum(1)   
        ones = torch.ones(size=pos_sums.size())    
        norm = torch.where(pos_sums > 0, pos_sums, ones.to(pos.device))   
        mean_log_prob_pos = (pos * log_prob).sum(1) / norm   # normalize by positives
        # print('\npositives: {} \nnegatives {}'.format(pos.sum(1), neg.sum(1)))
        # print('mean_log_prob_pos ', has_inf_or_nan(mean_log_prob_pos))
        loss = - mean_log_prob_pos
        # loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        loss = loss.mean()
        # print('loss.mean() ', has_inf_or_nan(loss))
        # print('loss {}'.format(loss))
        if has_inf_or_nan(loss)[0] or has_inf_or_nan(loss)[1]:
            print('\n inf found in loss with Positives {} and Negatives {}'.format(pos.sum(1), neg.sum(1)))
        return loss
    
        
    def cm_contrastive_loss(self, feats_c, feats_o, labels):  
        """
        input:  feats T-C-V
                      T: classes/anchors in BATCH (with repetition)
                      C: feature space dimensionality
                      V: views_per_class/anchor, i.e. samples from each class/anchor
                labels T
        return: loss
        """      
        # prepare feats                                    
        feats_c = torch.nn.functional.normalize(feats_c, p=2, dim=1)  
        feats_c = feats_c.transpose(dim0=1, dim1=2)  
        num_anchors, views_per_anchor, c = feats_c.shape  
        feats_c_flat = feats_c.contiguous().view(-1, c)  

        feats_o = torch.nn.functional.normalize(feats_o, p=2, dim=1) 
        feats_o = feats_o.transpose(dim0=1, dim1=2)  
        num_anchors, views_per_anchor, c = feats_o.shape 
        feats_o_flat = feats_o.contiguous().view(-1, c) 

        labels = labels.contiguous().view(-1, 1)  
        labels = labels.repeat(1, views_per_anchor)  
        labels = labels.view(-1, 1) 

        pos_mask, neg_mask = self.get_masks(labels, num_anchors, views_per_anchor) 
        dot_product = torch.div(torch.matmul(feats_c_flat, torch.transpose(feats_o_flat, 0, 1)), self.cross_modal_temperature)
        loss = self.InfoNCE_loss(pos_mask, neg_mask, dot_product) 
        return loss

    
