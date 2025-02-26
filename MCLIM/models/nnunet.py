import torch
import torch.nn as nn
from typing import List


class UNetEncoder(nn.Module):
    """
    This is a template for your custom ConvNet.
    It is required to implement the following three functions: `get_downsample_ratio`, `get_feature_map_channels`, `forward`.
    You can refer to the implementations in `pretrain\models\resnet.py` for an example.
    """
    def __init__(self, 
                 in_chans=1, 
                 depths=[2, 2, 2, 2, 1], 
                 dims=[32, 64, 128, 256, 320],
                 sparse=True,
                 ):
        super().__init__()

        self.stem = self._get_bottom_layer(in_chans, dims[0])
        self.stages = nn.ModuleList() 
        self.downsample_layers = nn.ModuleList()
        self.n_stages = len(depths)
        self.dims = dims
        
        for i in range(len(depths)):
            stage = nn.Sequential(
                *[self._get_bottom_layer(dims[i], dims[i]) for j in range(depths[i]-1)],
            )
            if i != len(depths) - 1:
                self.downsample_layers.append(self._get_down_layer(dims[i], dims[i+1]))

            self.stages.append(stage)

    def _get_bottom_layer(self, in_channels, out_channels):
        return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.InstanceNorm3d(out_channels),
                nn.PReLU(),
        )

    def _get_down_layer(self, in_channels, out_channels):
        return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm3d(out_channels),
                nn.PReLU(),
        )
    
    def get_downsample_ratio(self) -> int:
        """
        This func would ONLY be used in `SparseEncoder's __init__` (see `pretrain/encoder.py`).
        
        :return: the TOTAL downsample ratio of the ConvNet.
        E.g., for a ResNet-50, this should return 32.
        """
        return 2 ** (self.n_stages - 1)
    
    def get_feature_map_channels(self) -> List[int]:
        """
        This func would ONLY be used in `SparseEncoder's __init__` (see `pretrain/encoder.py`).
        
        :return: a list of the number of channels of each feature map.
        E.g., for a ResNet-50, this should return [256, 512, 1024, 2048].
        """
        return self.dims
    
    def forward(self, inp_bchwd: torch.Tensor, hierarchical=True):
        """
        The forward with `hierarchical=True` would ONLY be used in `SparseEncoder.forward` (see `pretrain/encoder.py`).
        
        :param inp_bchw: input image tensor, shape: (batch_size, channels, height, width).
        :param hierarchical: return the logits (not hierarchical), or the feature maps (hierarchical).
        :return:
            - hierarchical == False: return the logits of the classification task, shape: (batch_size, num_classes).
            - hierarchical == True: return a list of all feature maps, which should have the same length as the return value of `get_feature_map_channels`.
              E.g., for a ResNet-50, it should return a list [1st_feat_map, 2nd_feat_map, 3rd_feat_map, 4th_feat_map].
                    for an input size of 224, the shapes are [(B, 256, 56, 56), (B, 512, 28, 28), (B, 1024, 14, 14), (B, 2048, 7, 7)]
        """
        if hierarchical:
            x = self.stem(inp_bchwd)
            ls = []
            for i in range(self.n_stages):
                x = self.stages[i](x)             
                ls.append(x)
                if i != self.n_stages - 1:
                    x = self.downsample_layers[i](x)
            return ls
        else:
            raise NotImplementedError


class UNetDecoder(nn.Module):
    """
    This is a template for your custom ConvNet.
    It is required to implement the following three functions: `get_downsample_ratio`, `get_feature_map_channels`, `forward`.
    You can refer to the implementations in `pretrain\models\resnet.py` for an example.
    """
    def __init__(self, 
                 out_chans=1, 
                 depths=[2, 2, 2, 2, 0], 
                 dims=[32, 64, 128, 256, 320],
                 sparse=True,
                 ):
        super().__init__()

        self.stages = nn.ModuleList() 
        self.upsample_layers = nn.ModuleList()
        self.n_stages = len(depths)
        self.dims = dims
        self.width = dims
        
        for i in range(len(depths)-1, -1, -1):
            if depths[i] > 0:
                stage = nn.Sequential(
                    self._get_bottom_layer(dims[i] * 2, dims[i]),
                    *[self._get_bottom_layer(dims[i], dims[i]) for j in range(depths[i]-1)],
                )
                self.stages.append(stage)

            if i != 0:
                self.upsample_layers.append(self._get_up_layer(dims[i], dims[i-1]))   

        self.proj = nn.Conv3d(dims[0], out_chans, kernel_size=1, stride=1, bias=True)

    def _get_up_layer(self, in_channels, out_channels):
        return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)

    def _get_bottom_layer(self, in_channels, out_channels):
        return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.InstanceNorm3d(out_channels),
                nn.PReLU(),
        )
    
    def forward(self, to_dec: List[torch.Tensor]):
        to_dec.reverse()
        x = to_dec[0]
        for i, d in enumerate(self.stages):
            x = self.upsample_layers[i](x)
            if i + 1 < len(to_dec):
                x = torch.cat([x, to_dec[i+1]], dim=1)
            x = d(x)

        return self.proj(x)
    

class NSDecoder(nn.Module):
    """
    This is a template for your custom ConvNet.
    It is required to implement the following three functions: `get_downsample_ratio`, `get_feature_map_channels`, `forward`.
    You can refer to the implementations in `pretrain\models\resnet.py` for an example.
    """
    def __init__(self, 
                 out_chans=1, 
                 depths=[2, 2, 2, 2, 0], 
                 dims=[32, 64, 128, 256, 320],
                 sparse=True,
                 ):
        super().__init__()

        self.stages = nn.ModuleList() 
        self.upsample_layers = nn.ModuleList()
        self.n_stages = len(depths)
        self.dims = dims
        self.width = dims
        
        for i in range(len(depths)-1, -1, -1):
            if depths[i] > 0:
                stage = nn.Sequential(
                    self._get_bottom_layer(dims[i], dims[i]),
                    *[self._get_bottom_layer(dims[i], dims[i]) for j in range(depths[i]-1)],
                )
                self.stages.append(stage)

            if i != 0:
                self.upsample_layers.append(self._get_up_layer(dims[i], dims[i-1]))   

        self.proj = nn.Conv3d(dims[0], out_chans, kernel_size=1, stride=1, bias=True)

    def _get_up_layer(self, in_channels, out_channels):
        return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)

    def _get_bottom_layer(self, in_channels, out_channels):
        return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.InstanceNorm3d(out_channels),
                nn.PReLU(),
        )
    
    def forward(self, x: torch.Tensor):
        for i, d in enumerate(self.stages):
            x = self.upsample_layers[i](x)
            x = d(x)

        return self.proj(x)