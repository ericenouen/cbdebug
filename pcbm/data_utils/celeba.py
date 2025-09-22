
"""
Obtained from https://github.com/yewsiang/ConceptBottleneck and modified.
General utils for training, evaluation and data loading
"""
import os
import torch
import pickle
import numpy as np
import pandas as pd
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import BatchSampler
from torch.utils.data import Dataset, DataLoader

N_ATTRIBUTES=312

class CelebADataset(Dataset):
    """
    Returns a compatible Torch Dataset object customized for the CelebADataset dataset
    """
    SPLITS = {
        'tr': 0,
        'va': 1,
        'te': 2
    }

    def __init__(self, split, metadata, root_dir, num_classes, transform=None):
        """
        Arguments:
        pkl_file_paths: list of full path to all the pkl data
        use_attr: whether to load the attributes (e.g. False for simple finetune)
        image_dir: default = 'images'. Will be append to the parent dir
        transform: whether to apply any special transformation. Default = None, i.e. use standard ImageNet preprocessing
        """

        df = pd.read_csv(metadata)
        df = df[df["split"] == (self.SPLITS[split])]

        self.idx = list(range(len(df)))
        self.x = df["filename"].astype(str).map(lambda x: os.path.join(root_dir, x)).tolist()
        self.y = df["y"].tolist()

        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        i = self.idx[index]
        x = Image.open(self.x[i]).convert("RGB")
        if self.transform is not None:
            x = self.transform(x)
        y = torch.tensor(self.y[i], dtype=torch.long)
        return x, y, i
        

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None):
        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices)

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):  # Note: for single attribute dataset
        return dataset.y[idx]

    def __iter__(self):
        idx = (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))
        return idx

    def __len__(self):
        return self.num_samples


def load_celeba_data(is_training, batch_size, image_dir='images', metadata_path="", resampling=False, resol=299,
             preprocess=None, n_classes=200):
    """
    Note: Inception needs (299,299,3) images with inputs scaled between -1 and 1
    Loads data with transformations applied, and upsample the minority class if there is class imbalance and weighted loss is not used
    NOTE: resampling is customized for first attribute only, so change sampler.py if necessary
    """
    # is_training = any(['train.pkl' in f for f in pkl_paths])
    if is_training:
        split = 'tr'
    else:
        split = 'te'

    dataset = CelebADataset(split, metadata_path, image_dir, n_classes, transform=preprocess)

    # if is_training:
    #     drop_last = True
    #     shuffle = True
    # else:
    drop_last = False
    shuffle = False
    if resampling:
        sampler = BatchSampler(ImbalancedDatasetSampler(dataset), batch_size=batch_size, drop_last=drop_last)
        loader = DataLoader(dataset, batch_sampler=sampler)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return loader


class CelebAConceptDataset:
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        # Return the length of the dataset
        return len(self.images)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


# def get_concept_dicts(metadata):
#     n_concepts = len(metadata[0]["attribute_label"])
#     concept_info = {c: {1: [], 0: []} for c in range(n_concepts)}
#     for im_data in metadata:
#         for c, label in enumerate(im_data["attribute_label"]):
#             print(c)
#             img_path = im_data["img_path"]            
#             idx = img_path.split('/').index('CUB_200_2011')
#             img_path = '/'.join([CUB_DATA_DIR] + img_path.split('/')[idx+1:])
#             concept_info[c][label].append(img_path)
#     return concept_info
