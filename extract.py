from monai.utils import first
from monai.transforms import(
    Compose,
    AddChanneld,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    Activations,
)

from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference

import os
import torch
import matplotlib.pyplot as plt
from glob import glob
import numpy as np

in_dir = '/home/graduate/celal/WORD-V0.1.0/'
model_dir = '/home/graduate/celal/WORD-V0.1.0/results1/'
train_loss = np.load(os.path.join(model_dir, 'loss_train.npy'))
train_metric = np.load(os.path.join(model_dir, 'metric_train.npy'))
test_loss = np.load(os.path.join(model_dir, 'loss_test.npy'))
test_metric = np.load(os.path.join(model_dir, 'metric_test.npy'))

plt.figure("Results", (12, 6))
plt.subplot(2, 2, 1)
plt.title("Train dice loss")
x = [i + 1 for i in range(len(train_loss))]
y = train_loss
plt.plot(x, y)

plt.subplot(2, 2, 2)
plt.title("Train metric dice")
x = [i + 1 for i in range(len(train_metric))]
y = train_metric
plt.plot(x, y)

plt.subplot(2, 2, 3)
plt.title("Test dice loss")
x = [i + 1 for i in range(len(test_loss))]
y = test_loss
plt.plot(x, y)

plt.subplot(2, 2, 4)
plt.title("Test metric dice")
x = [i + 1 for i in range(len(test_metric))]
y = test_metric
plt.plot(x, y)

plt.savefig("/home/graduate/celal/WORD-V0.1.0/test2/test.png")
path_train_volumes = sorted(glob(os.path.join(in_dir, "imagesTr", "*.nii.gz")))
path_train_segmentation = sorted(glob(os.path.join(in_dir, "labelsTr", "*.nii.gz")))

path_test_volumes = sorted(glob(os.path.join(in_dir, "imagesVal", "*.nii.gz")))
path_test_segmentation = sorted(glob(os.path.join(in_dir, "labelsVal", "*.nii.gz")))

train_files = [{"image": image_name, "label": label_name} for image_name, label_name in zip(path_train_volumes, path_train_segmentation)]
test_files = [{"image": image_name, "label": label_name} for image_name, label_name in zip(path_test_volumes, path_test_segmentation)]
test_files = test_files[0:9]
test_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.5,1.5,1.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200,b_min=0.0, b_max=1.0, clip=True), 
        CropForegroundd(keys=['image', 'label'], source_key='image'),
        Resized(keys=["image", "label"], spatial_size=[128,128,64]),   
        ToTensord(keys=["image", "label"]),
    ]
)
test_ds = Dataset(data=test_files, transform=test_transforms)
test_loader = DataLoader(test_ds, batch_size=1)
device = torch.device("cuda:0")

model = UNet(
    dimensions=3,
    in_channels=1,
    out_channels=17,
    channels=(16, 32, 64, 128, 256), 
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)

model.load_state_dict(torch.load(os.path.join(model_dir, "best_metric_model.pth")))
model.eval()
sw_batch_size = 4
roi_size = (128, 128, 64)
j = 0
with torch.no_grad():
    for test_patient in test_loader:
        t_volume = test_patient['image']
        t_segmentation = test_patient['label']

        test_outputs = sliding_window_inference(t_volume.to(device), roi_size, sw_batch_size, model)
        sigmoid_activation = Activations(sigmoid=True)
        test_outputs = sigmoid_activation(test_outputs)
        test_outputs = test_outputs > 0.53

        for i in range(32):
            plt.figure("check", (18, 6))
            plt.subplot(1, 3, 1)
            plt.title(f"image {i}")
            plt.imshow(test_patient["image"][0, 0, :, :, i], cmap="gray")
            plt.subplot(1, 3, 2)
            plt.title(f"label {i}")
            plt.imshow(test_patient["label"][0, 0, :, :, i] != 0)
            plt.subplot(1, 3, 3)
            plt.title(f"output {i}")
            plt.imshow(test_outputs.detach().cpu()[0, 1, :, :, i])
            plt.savefig(f"/home/graduate/celal/WORD-V0.1.0/test_all/image{j}_{i}")
    j+=1