import os
import torch
import open_clip
import numpy as np

from monai.data import CacheDataset, DataLoader, Dataset, DistributedSampler, PersistentDataset, load_decathlon_datalist
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImage,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRangePercentilesd,
    Spacingd,
    SpatialPadd,
    Resized,
    ToTensord,
)
from monai.transforms.transform import MapTransform
    

class UniqueLabeld(MapTransform):
    """
    Convert the data-format to the format that only has unique label value.
    """

    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        if "sub_atlas" not in d.keys() or "cort_atlas" not in d.keys():
            d['sub_atlas'] = d['seg'][:, 1]
            d['cort_atlas'] = d['seg'][:, 2]

        onehots = []
        for i in range(d["sub_atlas"].shape[0]):
            sub_label = np.unique(d["sub_atlas"][i]).astype(np.uint8)
            # remove index 0 if exists
            if sub_label[0] == 0:
                sub_label = sub_label[1:]
            cort_label = np.unique(d["cort_atlas"][i]).astype(np.uint8)
            # remove index 0 if exists
            if cort_label[0] == 0:
                cort_label = cort_label[1:]
            
            label = np.concatenate((sub_label + 47, cort_label - 1), axis=0)
            onehot = np.zeros((69))
            onehot[label] = 1
            onehots.append(onehot)

        onehots = np.array(onehots)
        d['label'] = onehots
        return d


class Mask_Origin_Img(MapTransform):
    """
    Mask the input image for MIM.
    """

    def __init__(self, keys, img_size, mask_ratio, patch_size, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.mask_ratio = mask_ratio
        self.img_size = img_size 
        self.patch_size = patch_size
        self.patch_num_per_dim = int(img_size//patch_size)
        self.len_keep = round(self.patch_num_per_dim * self.patch_num_per_dim * self.patch_num_per_dim * (1 - self.mask_ratio))

    def __call__(self, data):
        d = dict(data)
        if self.mask_ratio>0:
            f: int = self.patch_num_per_dim
            idx = np.random.rand(f * f * f).argsort()
            idx = idx[:self.len_keep]
            msk = np.array(list(range(f * f * f))) 
            msk = np.where(np.isin(msk,idx),1,0)
            img = d['image']
            mask = np.zeros_like(img)
            
            for i in range(self.patch_num_per_dim):
                for j in range(self.patch_num_per_dim):
                    for k in range(self.patch_num_per_dim):
                            mask[:,i*self.patch_size:(i+1)*self.patch_size,j*self.patch_size:(j+1)*self.patch_size,k*self.patch_size:(k+1)*self.patch_size] = msk[(i*self.patch_num_per_dim*self.patch_num_per_dim)+(j*self.patch_num_per_dim)+k]

            d['mask_image'] = img * mask
            d['mask'] = mask
        return d



from collections.abc import Sequence
def custom_list_data_collate(batch: Sequence):
    data = torch.cat([patch['image'] for item in batch for patch in item], dim=0)   
    label = torch.cat([patch['label'] for item in batch for patch in item], dim=0)
    # sub = torch.cat([patch['sub_atlas'] for item in batch for patch in item], dim=0)
    # cort = torch.cat([patch['cort_atlas'] for item in batch for patch in item], dim=0)
    # return {'image': data, 'label': label, 'sub_atlas': sub, 'cort_atlas': cort}
    return {'image': data, 'label': label}


def build_dataset_to_pretrain(dataset_path, input_size, mim_ratio, patch_size, inference=False) -> Dataset:
    
    tr_transforms = Compose(
        [
            LoadImaged(keys=["image"], image_only=True),
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityRangePercentilesd(keys=["image"], lower=1, upper=99, b_min=0, b_max=1),
            NormalizeIntensityd(keys=["image"]),
            RandCropByPosNegLabeld(
                keys=["image", "sub_atlas", "cort_atlas", "mask"],
                label_key="mask",
                spatial_size=(input_size, input_size, input_size),
                pos=1,
                neg=0.1,
                num_samples=4,
            ),
            UniqueLabeld(keys=["sub_atlas", "cort_atlas"]),
            Mask_Origin_Img(keys=["image"], img_size=input_size, mask_ratio=mim_ratio, patch_size=patch_size),
            # Resized(keys=["sub_atlas", "cort_atlas"], spatial_size=(input_size // 16, input_size // 16, input_size // 16), mode="nearest"),
            ToTensord(keys=["image", "label"], track_meta=False),
        ]
    )
    
    datalist = []
    with open(dataset_path, "r") as f:
        scan_list = f.readlines()

    # load MNI152 template
    mask = LoadImage(image_only=True, ensure_channel_first=True)('MCLIM/standard/MNI152_T1_1mm_brain_mask.nii.gz')
    sub = LoadImage(image_only=True, ensure_channel_first=True)('MCLIM/standard/HarvardOxford-sub-maxprob-thr50-1mm.nii.gz')
    cort = LoadImage(image_only=True, ensure_channel_first=True)('MCLIM/standard/HarvardOxford-cort-maxprob-thr50-1mm.nii.gz')
    seg = np.concatenate((mask, sub, cort), axis=0)
    seg = np.expand_dims(seg, axis=0)

    for scan in scan_list:
        if 'roi' not in scan and 'seg' not in scan:
            datalist.append({'image': scan.strip(), 'mask': mask, 'sub_atlas': sub, 'cort_atlas': cort, 'label': None})

    print('Dataset all training: number of data: {}'.format(len(datalist)))
    
    dataset_train = Dataset(data=datalist, transform=tr_transforms)
    
    return dataset_train


if __name__ == '__main__':
    dataset = build_dataset_to_pretrain('MCLIM/adni_affine_ss_full_list_clean.txt', 64)
    data_loader_train = DataLoader(
        dataset=dataset, batch_size=2, num_workers=4, pin_memory=False, persistent_workers=True, collate_fn=custom_list_data_collate
    )
    for data in data_loader_train:
        print(data['data'].shape)
        print(data['label'].shape)
        break