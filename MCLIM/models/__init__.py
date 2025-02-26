# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from timm.loss import SoftTargetCrossEntropy

from models.convnext import ConvNeXt
from models.resnet import ResNet
from models.unet import UNetEncoder
from unet_decoder import UNetDecoder
_import_resnets_for_timm_registration = (ResNet,)


# log more
def _ex_repr(self):
    return ', '.join(
        f'{k}=' + (f'{v:g}' if isinstance(v, float) else str(v))
        for k, v in vars(self).items()
        if not k.startswith('_') and k != 'training'
        and not isinstance(v, (torch.nn.Module, torch.Tensor))
    )
for clz in (torch.nn.CrossEntropyLoss, SoftTargetCrossEntropy):
    if hasattr(clz, 'extra_repr'):
        clz.extra_repr = _ex_repr
    else:
        clz.__repr__ = lambda self: f'{type(self).__name__}({_ex_repr(self)})'



def build_sparse_encoder(name: str, input_size: int, sbn=False, drop_path_rate=0.0, verbose=False, sparse=True):
    from encoder import SparseEncoder, DenseEncoder
    
    if name == 'unet':
        cnn = UNetEncoder()
    elif name == 'swintransformer':
        return None
    
    if sparse:
        return SparseEncoder(cnn, input_size=input_size, sbn=sbn, verbose=verbose)
    else:
        return DenseEncoder(cnn, input_size=input_size, sbn=sbn, verbose=verbose)
    

def build_decoder(name: str):
    if name == 'unet':
        return UNetDecoder()
    elif name == 'swintransformer':
        return None
