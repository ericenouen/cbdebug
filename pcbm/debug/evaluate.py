"""
Load in model with saved weights
Select desired retraining algorithm
Retrain for # retrain epochs
Save retrained model
"""
import argparse
import os, sys
import pickle
import numpy as np
import torch

from models import PosthocLinearCBM, get_model

import pandas as pd
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from datasets import get_dataset_class
from eval_helper import eval_metrics


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone-dir", required=True, type=str, help="Folder where backbone is saved.")
    parser.add_argument("--dataset", default="cub", type=str)
    parser.add_argument("--backbone-name", default="resnet18_cub", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--alpha", default=0.99, type=float, help="Sparsity coefficient for elastic net.")
    parser.add_argument("--lr", default=1e-3, type=float)
    return parser.parse_args()


def main(args, backbone, preprocess):

    algorithms = ["Remove", "Retrain", "Augment", "PermutationWeighting", "CBDebug"]
    for a in algorithms:
        avg_acc = []
        worstgroup_acc = []
        for seed in range(1, 4):
            posthoc_layer = torch.load(f"concept_banks/llm_run_{seed}/pcbm_{a}_{args.dataset}__clip:ViT-L-14__multimodal_concept_clip:ViT-L-14_{args.dataset}__lam:0.02__alpha:0.99__seed:{(seed-1) % 3 + 1}.ckpt").float()
            posthoc_layer = posthoc_layer.to(args.device)

            dataset_class = get_dataset_class(args.dataset)
            split_names = ['tr', 'te']
            datasets = [dataset_class(split, None, transform=preprocess) for split in split_names]
            final_eval_loaders = [DataLoader(
                dataset=dset,
                batch_size=max(128, args.batch_size * 2),
                num_workers=args.num_workers)
                for dset in datasets
            ]
            
            posthoc_layer.predict = lambda x: posthoc_layer.forward(backbone.encode_image(x).float())

            ava, wga = evaluate_model(posthoc_layer, split_names, final_eval_loaders, args.device)
            avg_acc.append(ava)
            worstgroup_acc.append(wga)
        
        print(f"{a}:")
        print(avg_acc)
        print(worstgroup_acc)
        print(f"Average Accuracy: Mean = {100*np.mean(avg_acc):.4f}, Std = {100*np.std(avg_acc):.4f}")
        print(f"Worst Group Accuracy: Mean = {100*np.mean(worstgroup_acc):.4f}, Std = {100*np.std(worstgroup_acc):.4f}")

def evaluate_model(IMLModel, split_names, final_eval_loaders, device):
    final_results = {split: eval_metrics(IMLModel, loader, device)
                     for split, loader in zip(split_names, final_eval_loaders)}

    print(f"\tmean:\t[{final_results[split_names[1]]['overall']['accuracy']:.3f}]\n"
          f"\tworst:\t[{final_results[split_names[1]]['min_group']['accuracy']:.3f}]")
    print(f"\tf1:\t[{final_results[split_names[1]]['overall']['macro_avg']['f1-score']:.3f}]\n")
    print("Group-wise accuracy:")
    for split in final_results.keys():
        print('\t[{}] group-wise {}'.format(
            split, (np.array2string(
                pd.DataFrame(final_results[split]['per_group']).T['accuracy'].values,
                separator=', ', formatter={'float_kind': lambda x: "%.3f" % x}))))
    return final_results[split_names[1]]['overall']['accuracy'], final_results[split_names[1]]['min_group']['accuracy']

if __name__ == "__main__":
    args = config()
    # Get the backbone from the model zoo.
    backbone, preprocess = get_model(args, backbone_name=args.backbone_name)
    backbone = backbone.to(args.device)
    backbone.eval()
    main(args, backbone, preprocess)
