import torch
import os
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
    Resized
)
from monai.transforms.spatial.functional import resize
from monai.apps import DecathlonDataset

from monai.data import DataLoader, decollate_batch


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
        return d


class DownSample(MapTransform):
    def __call__(self, data):
        d = dict(data)
        img = d["image"]
        label = d["label"]
        align_corners=True
        dtype= torch.float32
        input_ndim=3
        anti_aliasing=True
        anti_aliasing_sigma=1
        lazy=True
        transform_info={}
        out_size = (64, 64, 32)
        output_modalities = []
        for i in range(img.shape[0]):
          modality = img[i].unsqueeze(0)
          resized_modality = resize(
              img=modality,
              out_size=out_size,
              mode='trilinear',
              align_corners=False,
              dtype=dtype,
              input_ndim=input_ndim,
              anti_aliasing=anti_aliasing,
              anti_aliasing_sigma=anti_aliasing_sigma,
              lazy=False,
              transform_info=transform_info
          )
          output_modalities.append(resized_modality)
        img = torch.cat(output_modalities, dim=0)

        output_labels = []
        for i in range(label.shape[0]):
          modality = label[i].unsqueeze(0)
          resized_modality = resize(
              img=modality,
              out_size=out_size,
              mode='nearest',
              dtype=dtype,
              align_corners=None,
              input_ndim=input_ndim,
              anti_aliasing=anti_aliasing,
              anti_aliasing_sigma=anti_aliasing_sigma,
              lazy=False,
              transform_info=transform_info
          )
          output_labels.append(resized_modality)
        label = torch.cat(output_labels, dim=0)

        d["image"] = img
        d["label"] = label
        return d


# Define transforms
train_transform = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        RandSpatialCropd(keys=["image", "label"], roi_size=[224, 224, 144], random_size=False),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    ]
)

val_transform = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        RandSpatialCropd(keys=["image", "label"], roi_size=[224, 224, 144], random_size=False),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
)

def get_data_loader(download=True, batch_size=1):
    
    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    if directory is not None:
        os.makedirs(directory, exist_ok=True)

    train_ds = DecathlonDataset(
        root_dir=directory,
        task="Task01_BrainTumour",
        transform=train_transform,
        section="training",
        download=download,
        cache_rate=0.0,
        num_workers=0,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    val_ds = DecathlonDataset(
        root_dir=directory,
        task="Task01_BrainTumour",
        transform=val_transform,
        section="validation",
        download=False,
        cache_rate=0.0,
        num_workers=0,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader