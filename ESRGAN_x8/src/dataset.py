import os
import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SRDataset(Dataset):

    def __init__(self, root_dir, resize=False):
        self.resize = resize
        super().__init__()

        if self.resize:
            self.lr_transforms = transforms.Compose([
                transforms.Resize(16),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0], std=[1])
            ])
            self.hr_transforms = transforms.Compose([
                transforms.Resize(128),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            self.lr_transforms = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0], std=[1])
            ])
            self.hr_transforms = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])

        PARENT_DIR = []
        for root, dirs, files in os.walk(root_dir):
            for d in dirs:
                PARENT_DIR.append(d)

        lr_path = [f for f in PARENT_DIR if "LR" in f][0]
        hr_path = [f for f in PARENT_DIR if "HR" in f][0]

        lr_path = os.path.realpath(os.path.join(root_dir, lr_path))
        hr_path = os.path.realpath(os.path.join(root_dir, hr_path))

        self.lr_files = sorted(glob.glob(lr_path + "/*.*"))
        self.hr_files = sorted(glob.glob(hr_path + "/*.*"))

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, index):
        hr = Image.open(self.hr_files[index])
        lr = Image.open(self.lr_files[index])

        hr = self.hr_transforms(hr)
        lr = self.lr_transforms(lr)

        return lr, hr


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    data = SRDataset(root_dir="../../training_images_x8", resize=False)
    idx = np.random.randint(len(data))
    LR, HR = data[idx]
    print(LR.shape)
    print(HR.shape)
    plt.imshow(HR.numpy().transpose(1, 2, 0), cmap="gray")
    plt.show()
