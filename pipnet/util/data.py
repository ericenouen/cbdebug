
import numpy as np
import argparse
import torch
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Dict
from torch import Tensor
import random
import pandas as pd
import os
from PIL import Image

PATH_TO_DATA = "../data"

def get_data(args: argparse.Namespace): 
    """
    Load the proper dataset based on the parsed arguments
    """
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset =='Waterbirds':
        return get_waterbirds(True, f'{PATH_TO_DATA}/waterbirds/waterbird_complete95_forest2water2', f'{PATH_TO_DATA}/waterbirds/metadata_waterbirds.csv', args.image_size, args.seed)
    if args.dataset =='MetaShift':
        return get_metashift(True, f'{PATH_TO_DATA}/MetaShiftCoco', f'{PATH_TO_DATA}/MetaShiftCoco/metadata_metashift.csv', args.image_size, args.seed)
    if args.dataset =='CelebA':
        return get_celeba(True, f'{PATH_TO_DATA}/celeba/img_align_celeba', f'{PATH_TO_DATA}/celeba/metadata_celeba.csv', args.image_size, args.seed)
    raise Exception(f'Could not load data set, data set "{args.dataset}" not found!')

def get_dataloaders(args: argparse.Namespace, device):
    """
    Get data loaders
    """
    # Obtain the dataset
    trainset, trainset_pretraining, trainset_normal, trainset_normal_augment, projectset, testset, testset_projection, classes, num_channels, train_indices, targets = get_data(args)
    
    # Determine if GPU should be used
    cuda = not args.disable_cuda and torch.cuda.is_available()
    to_shuffle = True
    sampler = None
    
    num_workers = args.num_workers
    
    if args.weighted_loss:
        if targets is None:
            raise ValueError("Weighted loss not implemented for this dataset. Targets should be restructured")
        # https://discuss.pytorch.org/t/dataloader-using-subsetrandomsampler-and-weightedrandomsampler-at-the-same-time/29907
        class_sample_count = torch.tensor([(targets[train_indices] == t).sum() for t in torch.unique(targets, sorted=True)])
        weight = 1. / class_sample_count.float()
        print("Weights for weighted sampler: ", weight, flush=True)
        samples_weight = torch.tensor([weight[t] for t in targets[train_indices]])
        # Create sampler, dataset, loader
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight),replacement=True)
        to_shuffle = False

    pretrain_batchsize = args.batch_size_pretrain 
    
    
    trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_size=args.batch_size,
                                            shuffle=to_shuffle,
                                            sampler=sampler,
                                            pin_memory=cuda,
                                            num_workers=num_workers,
                                            worker_init_fn=np.random.seed(args.seed),
                                            drop_last=True
                                            )
    if trainset_pretraining is not None:
        trainloader_pretraining = torch.utils.data.DataLoader(trainset_pretraining,
                                            batch_size=pretrain_batchsize,
                                            shuffle=to_shuffle,
                                            sampler=sampler,
                                            pin_memory=cuda,
                                            num_workers=num_workers,
                                            worker_init_fn=np.random.seed(args.seed),
                                            drop_last=True
                                            )
                                        
    else:        
        trainloader_pretraining = torch.utils.data.DataLoader(trainset,
                                            batch_size=pretrain_batchsize,
                                            shuffle=to_shuffle,
                                            sampler=sampler,
                                            pin_memory=cuda,
                                            num_workers=num_workers,
                                            worker_init_fn=np.random.seed(args.seed),
                                            drop_last=True
                                            )

    trainloader_normal = torch.utils.data.DataLoader(trainset_normal,
                                            batch_size=args.batch_size,
                                            shuffle=to_shuffle,
                                            sampler=sampler,
                                            pin_memory=cuda,
                                            num_workers=num_workers,
                                            worker_init_fn=np.random.seed(args.seed),
                                            drop_last=True
                                            )
    trainloader_normal_augment = torch.utils.data.DataLoader(trainset_normal_augment,
                                            batch_size=args.batch_size,
                                            shuffle=to_shuffle,
                                            sampler=sampler,
                                            pin_memory=cuda,
                                            num_workers=num_workers,
                                            worker_init_fn=np.random.seed(args.seed),
                                            drop_last=True
                                            )
    
    projectloader = torch.utils.data.DataLoader(projectset,
                                              batch_size = 1,
                                              shuffle=False,
                                              pin_memory=cuda,
                                              num_workers=num_workers,
                                              worker_init_fn=np.random.seed(args.seed),
                                              drop_last=False
                                              )
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=args.batch_size,
                                             shuffle=True, 
                                             pin_memory=cuda,
                                             num_workers=num_workers,
                                             worker_init_fn=np.random.seed(args.seed),
                                             drop_last=False
                                             )
    test_projectloader = torch.utils.data.DataLoader(testset_projection,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=cuda,
                                             num_workers=num_workers,
                                             worker_init_fn=np.random.seed(args.seed),
                                             drop_last=False
                                             )
    print("Num classes (k) = ", len(classes), classes[:5], "etc.", flush=True)
    return trainloader, trainloader_pretraining, trainloader_normal, trainloader_normal_augment, projectloader, testloader, test_projectloader, classes

class PIPDataset:
    SPLITS = {
        'tr': 0,
        'va': 1,
        'te': 2
    }
    def __init__(self, split, metadata_path, root_dir, transform=None):
        df = pd.read_csv(metadata_path)
        df = df[df["split"] == (self.SPLITS[split])]

        self.idx = list(range(len(df)))
        self.x = df["filename"].astype(str).map(lambda x: os.path.join(root_dir, x)).tolist()
        self.y = df["y"].tolist()
        self.imgs = list(zip(self.x, self.y))
        self.transform = transform
        self.classes = list(range(len(set(self.y))))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        i = self.idx[index]
        x = Image.open(self.x[i]).convert("RGB")
        if self.transform is not None:
            x = self.transform(x)
        y = torch.tensor(self.y[i], dtype=torch.long)
        return x, y

def create_datasets(transform1, transform2, transform_no_augment, num_channels:int, root_dir:str, metadata_path:str, seed:int, transform1p=None):

    trainvalset = PIPDataset('tr', metadata_path, root_dir)
    targets = trainvalset.y
    classes = trainvalset.classes
    indices = list(range(len(trainvalset)))

    train_indices = indices

    testset = PIPDataset('te', metadata_path, root_dir, transform=transform_no_augment)
    testset_projection = testset
    
    trainset = torch.utils.data.Subset(TwoAugSupervisedDataset(trainvalset, transform1=transform1, transform2=transform2), indices=train_indices)
    trainset_normal = torch.utils.data.Subset(PIPDataset('tr', metadata_path, root_dir, transform=transform_no_augment), indices=train_indices)
    trainset_normal_augment = torch.utils.data.Subset(PIPDataset('tr', metadata_path, root_dir, transform=transforms.Compose([transform1, transform2])), indices=train_indices)
    projectset = PIPDataset('tr', metadata_path, root_dir, transform=transform_no_augment)

    trainvalset_pr = PIPDataset('tr', metadata_path, root_dir)
    # targets_pr = trainvalset_pr.targets
    indices_pr = list(range(len(trainvalset_pr)))
    train_indices_pr = indices_pr
    # if test_dir is None:
    #     subset_targets_pr = list(np.array(targets_pr)[indices_pr])
    #     train_indices_pr, test_indices_pr = train_test_split(indices_pr,test_size=validation_size,stratify=subset_targets_pr, random_state=seed)

    trainset_pretraining = torch.utils.data.Subset(TwoAugSupervisedDataset(trainvalset_pr, transform1=transform1p, transform2=transform2), indices=train_indices_pr)
    
    return trainset, trainset_pretraining, trainset_normal, trainset_normal_augment, projectset, testset, testset_projection, classes, num_channels, train_indices, torch.LongTensor(targets)

def get_waterbirds(augment: bool, root_dir:str, metadata_path:str, img_size: int, seed:int):
    shape = (3, img_size, img_size)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean,std=std)
    transform_no_augment = transforms.Compose([
                            transforms.Resize(size=(img_size, img_size)),
                            transforms.ToTensor(),
                            normalize
                        ])
    transform1p = None
    if augment:
        transform1 = transforms.Compose([
            transforms.Resize(size=(img_size+8, img_size+8)), 
            TrivialAugmentWideNoColor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(img_size+4, scale=(0.95, 1.))
        ])
        transform1p = transforms.Compose([
            transforms.Resize(size=(img_size+32, img_size+32)), #for pretraining, crop can be bigger since it doesn't matter when bird is not fully visible
            TrivialAugmentWideNoColor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(img_size+4, scale=(0.95, 1.))
        ])
        transform2 = transforms.Compose([
                            TrivialAugmentWideNoShape(),
                            transforms.RandomCrop(size=(img_size, img_size)), #includes crop
                            transforms.ToTensor(),
                            normalize
                            ])
    else:
        transform1 = transform_no_augment
        transform2 = transform_no_augment
    
    return create_datasets(transform1, transform2, transform_no_augment, 3, root_dir, metadata_path, seed, transform1p)

def get_metashift(augment: bool, root_dir:str, metadata_path:str, img_size: int, seed:int):
    shape = (3, img_size, img_size)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean,std=std)
    transform_no_augment = transforms.Compose([
                            transforms.Resize(size=(img_size, img_size)),
                            transforms.ToTensor(),
                            normalize
                        ])
    transform1p = None
    if augment:
        transform1 = transforms.Compose([
            transforms.Resize(size=(img_size+8, img_size+8)), 
            TrivialAugmentWideNoColor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(img_size+4, scale=(0.95, 1.))
        ])
        transform1p = transforms.Compose([
            transforms.Resize(size=(img_size+32, img_size+32)),
            TrivialAugmentWideNoColor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(img_size+4, scale=(0.95, 1.))
        ])
        transform2 = transforms.Compose([
                            TrivialAugmentWideNoShape(),
                            transforms.RandomCrop(size=(img_size, img_size)), #includes crop
                            transforms.ToTensor(),
                            normalize
                            ])
    else:
        transform1 = transform_no_augment
        transform2 = transform_no_augment
    
    return create_datasets(transform1, transform2, transform_no_augment, 3, root_dir, metadata_path, seed, transform1p)

def get_celeba(augment: bool, root_dir:str, metadata_path:str, img_size: int, seed:int):
    shape = (3, img_size, img_size)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean,std=std)
    transform_no_augment = transforms.Compose([
                            transforms.Resize(size=(img_size, img_size)),
                            transforms.ToTensor(),
                            normalize
                        ])
    transform1p = None
    if augment:
        transform1 = transforms.Compose([
            transforms.Resize(size=(img_size+8, img_size+8)), 
            TrivialAugmentWideNoColor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(img_size+4, scale=(0.95, 1.))
        ])
        transform1p = transforms.Compose([
            transforms.Resize(size=(img_size+32, img_size+32)),
            TrivialAugmentWideNoColor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(img_size+4, scale=(0.95, 1.))
        ])
        transform2 = transforms.Compose([
                            TrivialAugmentWideNoColor(),
                            transforms.RandomCrop(size=(img_size, img_size)), #includes crop
                            transforms.ToTensor(),
                            normalize
                            ])
    else:
        transform1 = transform_no_augment
        transform2 = transform_no_augment
    
    return create_datasets(transform1, transform2, transform_no_augment, 3, root_dir, metadata_path, seed, transform1p)

class TwoAugSupervisedDataset(torch.utils.data.Dataset):
    r"""Returns two augmentation and no labels."""
    def __init__(self, dataset, transform1, transform2):
        self.dataset = dataset
        self.classes = dataset.classes
        self.targets = dataset.y
        self.imgs = list(zip(dataset.x, dataset.y))
        self.transform1 = transform1
        self.transform2 = transform2

    def __getitem__(self, index):
        image, target = self.dataset[index]
        image = self.transform1(image)
        return self.transform2(image), self.transform2(image), target, index

    def __len__(self):
        return len(self.dataset)

# function copied from https://pytorch.org/vision/stable/_modules/torchvision/transforms/autoaugment.html#TrivialAugmentWide (v0.12) and adapted
class TrivialAugmentWideNoColor(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.5, num_bins), True), 
            "ShearY": (torch.linspace(0.0, 0.5, num_bins), True), 
            "TranslateX": (torch.linspace(0.0, 16.0, num_bins), True), 
            "TranslateY": (torch.linspace(0.0, 16.0, num_bins), True), 
            "Rotate": (torch.linspace(0.0, 60.0, num_bins), True), 
        }

class TrivialAugmentWideNoShapeWithColor(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            "Identity": (torch.tensor(0.0), False),
            "Brightness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Color": (torch.linspace(0.0, 0.5, num_bins), True), 
            "Contrast": (torch.linspace(0.0, 0.5, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }

class TrivialAugmentWideNoShape(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            
            "Identity": (torch.tensor(0.0), False),
            "Brightness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Color": (torch.linspace(0.0, 0.02, num_bins), True), 
            "Contrast": (torch.linspace(0.0, 0.5, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }

