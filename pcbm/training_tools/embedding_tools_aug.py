import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset


def unpack_batch(batch):
    if len(batch) == 3:
        return batch[0], batch[1], batch[2]
    elif len(batch) == 2:
        return batch
    else:
        raise ValueError()


@torch.no_grad()
def get_aug_projections(args, algorithm, backbone, posthoc_layer, loader, train=True):
    all_projs, all_embs, all_lbls = None, None, None
    for batch in tqdm(loader):
        batch_X, batch_Y, indices = unpack_batch(batch)
        batch_X = batch_X.to(args.device)
        if train: # Augment only the training data
            batch_X = algorithm.augment(batch_X, batch_Y, indices)
        if "clip" in args.backbone_name:
            embeddings = backbone.encode_image(batch_X).detach().float()
        else:
            embeddings = backbone(batch_X).detach()
        projs = posthoc_layer.compute_dist(embeddings).detach().cpu().numpy()
        embeddings = embeddings.detach().cpu().numpy()
        if all_embs is None:
            all_embs = embeddings
            all_projs = projs
            all_lbls = batch_Y.numpy()
        else:
            all_embs = np.concatenate([all_embs, embeddings], axis=0)
            all_projs = np.concatenate([all_projs, projs], axis=0)
            all_lbls = np.concatenate([all_lbls, batch_Y.numpy()], axis=0)
    return all_embs, all_projs, all_lbls


class EmbDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        return x, y
    def __len__(self):
        return len(self.data)


def compute_projections(args, algorithm, backbone, posthoc_layer, train_loader, test_loader):
    # Get a clean conceptbank string
    # e.g. if the path is /../../cub_resnet-cub_0.1_100.pkl, then the conceptbank string is resnet-cub_0.1_100
    conceptbank_source = args.concept_bank.split("/")[-1].split(".")[0] 
    
    # To make it easier to analyize results/rerun with different params, we'll extract the embeddings and save them
    train_file = f"aug-train-embs_{args.dataset}__{args.backbone_name}__{conceptbank_source}.npy"
    test_file = f"aug-test-embs_{args.dataset}__{args.backbone_name}__{conceptbank_source}.npy"
    train_proj_file = f"aug-train-proj_{args.dataset}__{args.backbone_name}__{conceptbank_source}.npy"
    test_proj_file = f"aug-test-proj_{args.dataset}__{args.backbone_name}__{conceptbank_source}.npy"
    train_lbls_file = f"aug-train-lbls_{args.dataset}__{args.backbone_name}__{conceptbank_source}_lbls.npy"
    test_lbls_file = f"aug-test-lbls_{args.dataset}__{args.backbone_name}__{conceptbank_source}_lbls.npy"
    
    train_file = os.path.join(args.out_dir, train_file)
    test_file = os.path.join(args.out_dir, test_file)
    train_proj_file = os.path.join(args.out_dir, train_proj_file)
    test_proj_file = os.path.join(args.out_dir, test_proj_file)
    train_lbls_file = os.path.join(args.out_dir, train_lbls_file)
    test_lbls_file = os.path.join(args.out_dir, test_lbls_file)

    train_embs, train_projs, train_lbls = get_aug_projections(args, algorithm, backbone, posthoc_layer, train_loader, train=True)
    test_embs, test_projs, test_lbls = get_aug_projections(args, algorithm, backbone, posthoc_layer, test_loader, train=False)

    # np.save(train_file, train_embs)
    # np.save(test_file, test_embs)
    # np.save(train_proj_file, train_projs)
    # np.save(test_proj_file, test_projs)
    # np.save(train_lbls_file, train_lbls)
    # np.save(test_lbls_file, test_lbls)
    
    return train_embs, train_projs, train_lbls, test_embs, test_projs, test_lbls
