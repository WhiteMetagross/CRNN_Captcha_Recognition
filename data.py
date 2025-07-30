#This program is the data loading module for the CAPTCHA recognition model.
#It defines a dataset class that handles image loading, preprocessing, and augmentation.

import warnings
warnings.simplefilter("ignore", UserWarning)

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

#Define the character set used in the CAPTCHA images.
class CaptchaDataset(Dataset):
    def __init__(self, root_dir, charset, height, width, is_train=True):
        self.root_dir = root_dir
        self.charset = charset
        self.height = height
        self.width = width
        self.is_train = is_train
        self.image_paths, self.labels = self._load_data()
        self.transform = self._get_transform()

    #Load images and labels from the specified directory.
    #It scans the directory for image files, verifies them, and filters out corrupted files.
    def _load_data(self):
        image_paths = []
        labels = []
        corrupted_files = []
        valid_extensions = ('.png', '.jpg', '.jpeg')
        
        print(f"Scanning and verifying images in {self.root_dir}...")
        file_list = os.listdir(self.root_dir)
        for filename in tqdm(file_list, desc="Scanning dataset"):
            if filename.lower().endswith(valid_extensions):
                label = os.path.splitext(filename)[0]
                if all(c in self.charset for c in label) and label:
                    path = os.path.join(self.root_dir, filename)
                    try:
                        with Image.open(path) as img:
                            img.verify()
                        image_paths.append(path)
                        labels.append(label)
                    except Exception:
                        corrupted_files.append(path)

        if corrupted_files:
            print(f"Warning: Found {len(corrupted_files)} corrupted image files. They will be skipped.")
        
        print(f"Loaded {len(image_paths)} valid samples from {self.root_dir}")
        return image_paths, labels

    #Define the transformations to be applied to the images.
    #For training, it includes augmentations like rotation, noise addition, and color jittering
    def _get_transform(self):
        if self.is_train:
            return A.Compose([
                A.Resize(self.height, self.width),
                A.Affine(rotate=(-2, 2), translate_percent=0.05, scale=(0.95, 1.05), p=0.3),
                A.OneOf([
                    A.GaussNoise(var_limit=(5.0, 15.0), p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.3), p=1.0),
                ], p=0.3),
                A.OneOf([
                    A.MotionBlur(blur_limit=3, p=1.0),
                    A.GaussianBlur(blur_limit=3, p=1.0),
                ], p=0.2),
                A.ImageCompression(quality_lower=85, quality_upper=95, p=0.2),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.4),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.4),
                A.CoarseDropout(max_holes=3, max_height=8, max_width=8, fill_value=0, p=0.2),
                A.SafeRotate(limit=1, border_mode=0, p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(self.height, self.width),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            img = Image.open(path).convert('RGB')
            img = np.array(img)
            img = self.transform(image=img)['image']
            enc = [self.charset.index(c) for c in label]
            return img, torch.tensor(enc, dtype=torch.long)
        except Exception:
            return self.__getitem__((idx + 1) % len(self.image_paths))

#Collate function to handle variable length labels and stack images.
#It pads the labels to the maximum length in the batch and returns images, padded labels, and lengths.
def collate_fn(batch):
    imgs, lbls = zip(*batch)
    imgs = torch.stack(imgs)
    lengths = torch.IntTensor([len(l) for l in lbls])
    padded = torch.nn.utils.rnn.pad_sequence(lbls, batch_first=True, padding_value=0)
    return imgs, padded, lengths