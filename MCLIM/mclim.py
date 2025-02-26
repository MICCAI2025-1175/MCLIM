import math
import os
import sys
from pprint import pformat
from typing import List


import numpy as np
import torch
import torch.nn as nn
import open_clip
import nibabel as nib
from timm.models.layers import trunc_normal_
from torch.nn import functional as F
from torch.cuda.amp import autocast

from models.nnunetv2 import UNetEncoder, UNetDecoder

from transformers import AutoConfig
from transformers.models.bert.modeling_bert import *

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

class ClipLoss(nn.Module):
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=True,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale=20):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return total_loss
    

class MatchingLoss(nn.Module):
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

    def forward(self, image_features, text_features, labels, logit_scale):
        logits_per_image = logit_scale * image_features @ text_features.T
        total_loss = F.binary_cross_entropy_with_logits(logits_per_image, labels)
        # evaluate matching accuracy
        probabilities = torch.sigmoid(logits_per_image)
        predictions = (probabilities > 0.5).float()  # Apply threshold
        # matching_accuracy = (predictions == labels).float().mean()

        return total_loss

def mask_mse(input,recon,mask):
    mask = 1-mask
    input = input*mask
    recon = recon*mask
    scaler = torch.sum(mask[0,0,])
    bs = input.size()[0]
    loss = torch.sum(torch.square(input - recon))
    loss = loss/scaler
    loss = loss/bs
    return loss
    
class ReconstructLoss(nn.Module):
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod


    def forward(self, origin_image, reconstruct_image, mask):
        
        total_loss = mask_mse(origin_image, reconstruct_image, mask)
        return total_loss    

class MCLIM(nn.Module):
    def __init__(
            self,
            rank=0,
            world_size=1,
            local_batch_size=1
    ):
        super().__init__()
        self.local_bs = local_batch_size
        self.image_encoder = UNetEncoder(dims=[32, 64, 128, 256, 512])
        self.text_encoder = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')[0].text
        self.image_decoder = UNetDecoder(dims=[32, 64, 128, 256, 512])
        self.clip_loss1 = ClipLoss(rank=rank, world_size=world_size)
        self.clip_loss2 = MatchingLoss(rank=rank, world_size=world_size)
        self.reconstruct_loss = ReconstructLoss(rank=rank, world_size=world_size)
        self.register_buffer('text_tokens', torch.randn(69, 64).long())
        text_tokens = torch.load('The path of atlas_base_token', map_location='cpu').detach()
        self.text_tokens.copy_(text_tokens)
        self.scale_factor = nn.Parameter(torch.ones([]) * np.log(1 / 1))
    
    def forward(self, image, label, mask, mask_image, weight_recon):
        label = label.squeeze()
        # encode prototypes
        num_iters = self.text_tokens.shape[0] // self.local_bs
        encoded_text_tokens_list = []
        for i in range(num_iters):
            text_tokens_batch = self.text_tokens[i*self.local_bs:(i+1)*self.local_bs]
            encoded_text_tokens_batch, _ = self.text_encoder(text_tokens_batch)
            encoded_text_tokens_list.append(encoded_text_tokens_batch)
        # process remaining tokens
        if self.text_tokens.shape[0] % self.local_bs != 0:
            text_tokens_batch = self.text_tokens[num_iters*self.local_bs:]
            encoded_text_tokens_batch, _ = self.text_encoder(text_tokens_batch)
            encoded_text_tokens_list.append(encoded_text_tokens_batch)

        text_prototypes = torch.cat(encoded_text_tokens_list, dim=0)

        if mask_image is not None:
            image_features_pyramid = self.image_encoder(mask_image)
        else:
            image_features_pyramid = self.image_encoder(image)

        image_local_features = image_features_pyramid[-1]
        
        # Reconstruct Image
        if (mask_image is not None) and (weight_recon>0):
            reconstruct_img = self.image_decoder(image_features_pyramid)
            recon_loss = self.reconstruct_loss(image, reconstruct_img, mask)
        else:
            recon_loss = 0

        # 3D global average pooling
        image_global_features = image_local_features.mean(dim=(2, 3, 4))

        # grep text features according to label
        text_global_features = torch.einsum('bj,jc->bc', label.float(), text_prototypes)

        # normalized features
        image_global_features = image_global_features / image_global_features.norm(dim=1, keepdim=True)
        text_global_features = text_global_features / text_global_features.norm(dim=1, keepdim=True)
        logit_scale = self.scale_factor.exp()

        clip_loss1 = self.clip_loss1(image_global_features, text_global_features)
        clip_loss2 = self.clip_loss2(image_global_features, text_prototypes, label, logit_scale)
        total_loss = clip_loss1 + clip_loss2 + weight_recon*recon_loss
        

        return total_loss, clip_loss1, clip_loss2, recon_loss
        


if __name__ == '__main__':
    img = torch.randn(2, 1, 64, 64, 64)
