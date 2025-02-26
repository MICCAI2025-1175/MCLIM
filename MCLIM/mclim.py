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


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=True,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


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

class LocalCrossAttention(nn.Module):
    def __init__(self, embed_dim, drop_rate=0):
        super(LocalCrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.query1 = nn.Linear(embed_dim, embed_dim)
        self.key1 = nn.Linear(embed_dim, embed_dim)
        self.value1 = nn.Linear(embed_dim, embed_dim)
        self.dropout1 = nn.Dropout(drop_rate)
        self.query2 = nn.Linear(768, embed_dim)
        self.key2 = nn.Linear(768, embed_dim)
        self.value2 = nn.Linear(768, embed_dim)
        self.dropout2 = nn.Dropout(drop_rate)

    def forward(
        self,
        input_tensor1,
        input_tensor2,
        attention_mask1=None,
        attention_mask2=None      
    ):
        # reshape input tensor
        B = input_tensor1.size(0)
        L1 = input_tensor1.size(2)
        L2 = input_tensor2.size(2)
        input_tensor1 = input_tensor1.permute(0, 2, 1).reshape(B*L1, 512)
        input_tensor2 = input_tensor2.permute(0, 2, 1).reshape(B*L2, 768)

        # for vision input [B, I]
        query_layer1 = self.query1(input_tensor1)
        key_layer1 = self.key1(input_tensor1)
        value_layer1 = self.value1(input_tensor1)

        # for text input [B, T]
        query_layer2 = self.query2(input_tensor2)
        key_layer2 = self.key2(input_tensor2)
        value_layer2 = self.value2(input_tensor2)

        # reshape qkv
        query_layer1 = query_layer1.reshape(B, L1, self.embed_dim)
        key_layer1 = key_layer1.reshape(B, L1, self.embed_dim)
        value_layer1 = value_layer1.reshape(B, L1, self.embed_dim)
        query_layer2 = query_layer2.reshape(B, L2, self.embed_dim)
        key_layer2 = key_layer2.reshape(B, L2, self.embed_dim)
        value_layer2 = value_layer2.reshape(B, L2, self.embed_dim)

        attention_scores1 = torch.einsum('bif,bjf->bij', query_layer2, key_layer1) # [B, T, D] @ [B, D, I] = [B, T, I]
        attention_scores1 = attention_scores1 / math.sqrt(self.embed_dim)
        if attention_mask1 is not None:
            attention_scores1 = attention_scores1 + attention_mask1

        # Sigmoid is better in this case
        # TODO: pre-normalize vs. post-normalize 
        attention_probs1 = F.sigmoid(attention_scores1)
     
        attention_probs1 = self.dropout1(attention_probs1)
        context_layer1 = torch.einsum('bij,bjf->bif', attention_probs1, value_layer1) # [B, T, I] @ [B, I, D] = [B, T, D]
        attention_scores2 = torch.einsum('bif,bjf->bij', query_layer1, key_layer2) # [B, I, D] @ [B, D, T] = [B, I, T]
        attention_scores2 = attention_scores2 / math.sqrt(self.embed_dim)

        if attention_mask2 is not None:
            attention_scores2 = attention_scores2 + attention_mask2
       
        attention_probs2 = F.sigmoid(attention_scores2)

        attention_probs2 = self.dropout2(attention_probs2)
        context_layer2 = torch.einsum('bij,bjf->bif', attention_probs2, value_layer2) # [B, I, T] @ [B, T, D] = [B, I, D]
        return context_layer2, attention_probs2, context_layer1, attention_probs1


class CrossModalityBertDecoder(BertModel):
    def __init__(self, config=None):
        if config is None:
            config = AutoConfig.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")
        config.num_hidden_layers = 4
        # config.is_decoder = True # 
        # config.add_cross_attention = True
        super().__init__(config, False) # no pooling layer for sentence-wise decoder
    
    def forward(self, x, y):
        '''
        x: x
        y: image_embed
        '''
        # return super().forward(inputs_embeds=x, encoder_hidden_states=y, return_dict=True)['last_hidden_state']
        return super().forward(inputs_embeds=x, return_dict=True)['last_hidden_state']
    

class SpatialDropout(nn.Module):
    def __init__(self, p=0.5):
        super(SpatialDropout, self).__init__()
        self.p = p
        # self.fmap_size = 4 # 64
        self.fmap_size = 6 # 96
        self.len_keep = round(self.fmap_size * self.fmap_size * self.fmap_size * (1 - self.p))

    def mask(self, B: int, device, generator=None):
        f: int = self.fmap_size
        idx = torch.rand(B, f * f * f, generator=generator).argsort(dim=1)
        idx = idx[:, :self.len_keep].to(device)  # (B, len_keep)
        return torch.zeros(B, f * f * f, dtype=torch.bool, device=device).scatter_(dim=1, index=idx, value=True).view(B, 1, f, f, f)

    def forward(self, input):
        cur_active = self.mask(input.size(0), input.device)
        return cur_active
    
        # cur_active_o = cur_active.repeat_interleave(16, 2).repeat_interleave(16, 3)
    

class TextDropout(nn.Module):
    def __init__(self, p=0.5):
        super(TextDropout, self).__init__()
        self.p = p
        self.len_keep = round(256 * (1 - self.p))

    def mask(self, B: int, device, generator=None):
        idx = torch.rand(B, 256, generator=generator).argsort(dim=1)
        idx = idx[:, :self.len_keep].to(device)  # (B, len_keep)
        return torch.zeros(B, 256, dtype=torch.bool, device=device).scatter_(dim=1, index=idx, value=True).view(B, 256, 1)

    def forward(self, input):
        cur_active = self.mask(input.size(0), input.device)
        return cur_active
    

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
        # self.local_cross_attention = LocalCrossAttention(512)
        # self.local_text_projection =  nn.Linear(512, 768)
        self.clip_loss1 = ClipLoss(rank=rank, world_size=world_size)
        self.clip_loss2 = MatchingLoss(rank=rank, world_size=world_size)
        self.reconstruct_loss = ReconstructLoss(rank=rank, world_size=world_size)
        self.register_buffer('text_tokens', torch.randn(69, 64).long())
        text_tokens = torch.load('MCLIM/atlas_token_v2.pth', map_location='cpu').detach()
        self.text_tokens.copy_(text_tokens)
        self.scale_factor = nn.Parameter(torch.ones([]) * np.log(1 / 1))
        # self.image_dropout = SpatialDropout(0.75)
        # self.text_dropout = TextDropout(0.5)
        # self.mask_tokens = nn.ParameterList()
        # self.text_mask_token = nn.Parameter(torch.zeros(1, 1, 768))
        # trunc_normal_(self.text_mask_token, mean=0, std=.02, a=-.02, b=.02)
        # for c in [32, 64, 128, 256]:
        #     # create mask token
        #     p = nn.Parameter(torch.zeros(1, c, 1, 1, 1))
        #     trunc_normal_(p, mean=0, std=.02, a=-.02, b=.02)
        #     self.mask_tokens.append(p)
    
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
        # clip_loss2 = self.clip_loss2(image_global_features, text_prototypes, label, logit_scale)
        # total_loss = clip_loss1 + clip_loss2 + weight_recon*recon_loss
        total_loss =  clip_loss1 + weight_recon*recon_loss

        return total_loss, clip_loss1, 0, recon_loss
        


if __name__ == '__main__':
    img = torch.randn(2, 1, 64, 64, 64)
