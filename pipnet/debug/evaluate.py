"""
Load in model with saved weights
Select desired retraining algorithm
Retrain for # retrain epochs
Save retrained model
"""
from pipnet.pipnet import PIPNet, get_network
from util.log import Log
import torch.nn as nn
from util.args import get_args, save_args, get_optimizer_nn
from util.data import get_dataloaders
import torch
import sys, os
import random
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from datasets import get_dataset_class
from eval_helper import eval_metrics


def run_pipnet(args=None):

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    args = args or get_args()
    assert args.batch_size > 1

    # Create a logger
    log = Log(args.log_dir)
    print("Log dir: ", args.log_dir, flush=True)
    # Log the run arguments
    save_args(args, log.metadata_dir)
    
    gpu_list = args.gpu_ids.split(',')
    device_ids = []
    if args.gpu_ids!='':
        for m in range(len(gpu_list)):
            device_ids.append(int(gpu_list[m]))
    
    global device
    if not args.disable_cuda and torch.cuda.is_available():
        if len(device_ids)==1:
            device = torch.device('cuda:{}'.format(args.gpu_ids))
        elif len(device_ids)==0:
            device = torch.device('cuda')
            print("CUDA device set without id specification", flush=True)
            device_ids.append(torch.cuda.current_device())
        else:
            print("This code should work with multiple GPU's but we didn't test that, so we recommend to use only 1 GPU.", flush=True)
            device_str = ''
            for d in device_ids:
                device_str+=str(d)
                device_str+=","
            device = torch.device('cuda:'+str(device_ids[0]))
    else:
        device = torch.device('cpu')
     
    # Log which device was actually used
    print("Device used: ", device, "with id", device_ids, flush=True)
    
    # Obtain the classes
    classes = get_dataloaders(args, device)[7]
    
    # Create a convolutional network based on arguments and add 1x1 conv layer
    feature_net, add_on_layers, pool_layer, classification_layer, num_prototypes = get_network(len(classes), args)
   
    # Create a PIP-Net
    net = PIPNet(num_classes=len(classes),
                    num_prototypes=num_prototypes,
                    feature_net = feature_net,
                    args = args,
                    add_on_layers = add_on_layers,
                    pool_layer = pool_layer,
                    classification_layer = classification_layer
                    )
    net = net.to(device=device)
    net = nn.DataParallel(net, device_ids = device_ids)

    means = []
    stds = []
    algorithms = ["Remove", "Retrain", "Augment", "ProtoPDebug", "PermutationWeighting", "CBDebug"]
    for a in algorithms:
        avg_acc = []
        worstgroup_acc = []
        for seed in range(1, 7):
            with torch.no_grad():
                if args.state_dict_dir_net != '':
                    checkpoint = torch.load(f"{args.state_dict_dir_net}/run_{seed}/pipnet_{args.dataset.lower()}_finetune_{a}/checkpoints/net_trained_{a}",map_location=device)
                    net.load_state_dict(checkpoint['model_state_dict'],strict=True) 
                    print("Pretrained network loaded", flush=True)
                    net.module._multiplier.requires_grad = False
            
            dataset_class = get_dataset_class(args.dataset)
            split_names = ['te']
            datasets = [dataset_class(split, None) for split in split_names]
            final_eval_loaders = [DataLoader(
                dataset=dset,
                batch_size=max(128, args.batch_size * 2),
                num_workers=args.num_workers)
                for dset in datasets
            ]
            from functools import partial
            # Add a predict function because evaluation script expects one
            net.predict = lambda *args, **kwargs: partial(net.forward, inference=True)(*args, **kwargs)[2]

            ava, wga = evaluate_model(net, split_names, final_eval_loaders, device)
            avg_acc.append(ava)
            worstgroup_acc.append(wga)

        print(f"Algorithm: {a}")
        print(f"Average Accuracy: Mean = {100*np.mean(avg_acc):.4f}, Std = {100*np.std(avg_acc):.4f}")
        print(f"Worst Group Accuracy: Mean = {100*np.mean(worstgroup_acc):.4f}, Std = {100*np.std(worstgroup_acc):.4f}")
        means.append(np.mean(worstgroup_acc))
        stds.append(np.std(worstgroup_acc))
            
def evaluate_model(IMLModel, split_names, final_eval_loaders, device):
    final_results = {split: eval_metrics(IMLModel, loader, device)
                     for split, loader in zip(split_names, final_eval_loaders)}

    # print(f"\tmean:\t[{final_results['te']['overall']['accuracy']:.3f}]\n"
    #       f"\tworst:\t[{final_results['te']['min_group']['accuracy']:.3f}]")
    # print(f"\tf1:\t[{final_results['te']['overall']['macro_avg']['f1-score']:.3f}]\n")
    # print("Group-wise accuracy:")
    # for split in final_results.keys():
    #     print('\t[{}] group-wise {}'.format(
    #         split, (np.array2string(
    #             pd.DataFrame(final_results[split]['per_group']).T['accuracy'].values,
    #             separator=', ', formatter={'float_kind': lambda x: "%.3f" % x}))))
    return final_results['te']['overall']['accuracy'], final_results['te']['min_group']['accuracy']

if __name__ == '__main__':
    args = get_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    print_dir = os.path.join(args.log_dir,'out.txt')
    tqdm_dir = os.path.join(args.log_dir,'tqdm.txt')
    if not os.path.isdir(args.log_dir):
        os.mkdir(args.log_dir)
    
    sys.stdout.close()
    sys.stderr.close()
    sys.stdout = open(print_dir, 'w')
    sys.stderr = open(tqdm_dir, 'w')
    run_pipnet(args)
    
    sys.stdout.close()
    sys.stderr.close()