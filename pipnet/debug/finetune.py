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
from debug.retrain import train_pipnet
from pipnet.test import eval_pipnet
from util.visualize_prediction import vis_pred
import torch
import sys, os
import random
import numpy as np
import matplotlib.pyplot as plt
from util.vis_pipnet import visualize, visualize_topk
import wandb

from debug.algorithm import get_algorithm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from datasets import get_dataset_class
from eval_helper import eval_metrics
from torch.utils.data import DataLoader
import pandas as pd
from copy import deepcopy

def run_pipnet(args=None):

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    wandb.init(project="cbdebug", config=args)

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
    
    # Obtain the dataset and dataloaders
    trainloader, trainloader_pretraining, trainloader_normal, trainloader_normal_augment, projectloader, testloader, test_projectloader, classes = get_dataloaders(args, device)
    if len(classes)<=20:
        if args.validation_size == 0.:
            print("Classes: ", testloader.dataset.class_to_idx, flush=True)
        else:
            print("Classes: ", str(classes), flush=True)
    
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
    
    optimizer_net, optimizer_classifier, params_to_freeze, params_to_train, params_backbone = get_optimizer_nn(net, args)   

    # Load model
    with torch.no_grad():
        if args.state_dict_dir_net != '':
            epoch = 0
            checkpoint = torch.load(args.state_dict_dir_net,map_location=device)
            net.load_state_dict(checkpoint['model_state_dict'],strict=True) 
            print("Pretrained network loaded", flush=True)
            net.module._multiplier.requires_grad = False
            try:
                optimizer_net.load_state_dict(checkpoint['optimizer_net_state_dict']) 
            except:
                pass
            # if torch.mean(net.module._classification.weight).item() > 1.0 and torch.mean(net.module._classification.weight).item() < 3.0 and torch.count_nonzero(torch.relu(net.module._classification.weight-1e-5)).float().item() > 0.8*(num_prototypes*len(classes)): #assume that the linear classification layer is not yet trained (e.g. when loading a pretrained backbone only)
            #     print("We assume that the classification layer is not yet trained. We re-initialize it...", flush=True)
            #     torch.nn.init.normal_(net.module._classification.weight, mean=1.0,std=0.1) 
            #     torch.nn.init.constant_(net.module._multiplier, val=2.)
            #     print("Classification layer initialized with mean", torch.mean(net.module._classification.weight).item(), flush=True)
            #     if args.bias:
            #         torch.nn.init.constant_(net.module._classification.bias, val=0.)
            # else: #uncomment these lines if you want to load the optimizer too
            if 'optimizer_classifier_state_dict' in checkpoint.keys():
                optimizer_classifier.load_state_dict(checkpoint['optimizer_classifier_state_dict'])
            
        else:
            print("Must load model for finetuning")
            
    tr_wga, test_wga = evaluate_model(net, args, device)
    wandb.log({"test WGA": test_wga})
    wandb.log({"train WGA": tr_wga})
            
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from datasets import get_dataset_class
    dataset_class = get_dataset_class(args.dataset)

    algorithm = get_algorithm(args)
    algorithm.dataset = dataset_class('tr', None)
    algorithm.initialize(net, trainloader)
    # for i in range(net.module._classification.weight.shape[0]):
    #     print(net.module._classification.weight[i])
    # print(torch.sum(net.module._classification.weight > 0))
    if args.algorithm == "Remove":
        net = algorithm.remove(net)
        tr_wga, test_wga = evaluate_model(net, args, device)
        wandb.log({"test WGA": test_wga})
        wandb.log({"train WGA": tr_wga})
        torch.save({'model_state_dict': net.state_dict()}, os.path.join(os.path.join(args.log_dir, 'checkpoints'), 'net_trained_Remove'))
        return
    else:
        net = algorithm.reinitialize(net)
        net = algorithm.remove(net)
    tr_wga, test_wga = evaluate_model(net, args, device)
    wandb.log({"test WGA": test_wga})
    wandb.log({"train WGA": tr_wga})
    for i in range(net.module._classification.weight.shape[0]):
        print(net.module._classification.weight[i])

    # Define classification loss function and scheduler
    criterion = nn.NLLLoss(reduction='mean').to(device)
    # scheduler_net = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_net, T_max=len(trainloader_pretraining)*args.epochs_pretrain, eta_min=args.lr_block/100., last_epoch=-1)

    # Forward one batch through the backbone to get the latent output size
    with torch.no_grad():
        xs1, _, _, _ = next(iter(trainloader))
        xs1 = xs1.to(device)
        proto_features, _, _ = net(xs1)
        wshape = proto_features.shape[-1]
        args.wshape = wshape #needed for calculating image patch size
        print("Output shape: ", proto_features.shape, flush=True)
    
    if net.module._num_classes == 2:
        # Create a csv log for storing the test accuracy, F1-score, mean train accuracy and mean loss for each epoch
        log.create_log('log_epoch_overview', 'epoch', 'test_top1_acc', 'test_f1', 'almost_sim_nonzeros', 'local_size_all_classes','almost_nonzeros_pooled', 'num_nonzero_prototypes', 'mean_train_acc', 'mean_train_loss_during_epoch')
        print("Your dataset only has two classes. Is the number of samples per class similar? If the data is imbalanced, we recommend to use the --weighted_loss flag to account for the imbalance.", flush=True)
    else:
        # Create a csv log for storing the test accuracy (top 1 and top 5), mean train accuracy and mean loss for each epoch
        log.create_log('log_epoch_overview', 'epoch', 'test_top1_acc', 'test_top5_acc', 'almost_sim_nonzeros', 'local_size_all_classes','almost_nonzeros_pooled', 'num_nonzero_prototypes', 'mean_train_acc', 'mean_train_loss_during_epoch')
    
    
    # SECOND TRAINING PHASE
    # re-initialize optimizers and schedulers for second training phase
    optimizer_net, optimizer_classifier, params_to_freeze, params_to_train, params_backbone = get_optimizer_nn(net, args)            
    # scheduler_net = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_net, T_max=len(trainloader)*args.epochs, eta_min=args.lr_net/100.)
    scheduler_net = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_net, T_max=len(trainloader)*args.epochs, eta_min=args.lr_net/1.)

    # scheduler for the classification layer is with restarts, such that the model can re-active zeroed-out prototypes. Hence an intuitive choice. 
    if args.epochs<=30:
        scheduler_classifier = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_classifier, T_0=5, eta_min=0.001, T_mult=1, verbose=False)
    else:
        scheduler_classifier = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_classifier, T_0=10, eta_min=0.001, T_mult=1, verbose=False)
    for param in net.module.parameters():
        param.requires_grad = False
    for param in net.module._classification.parameters():
        param.requires_grad = True
    
    frozen = True
    lrs_net = []
    lrs_classifier = []
    best_wga = 0
   
    for epoch in range(1, args.epochs + 1):
        epochs_to_finetune = 1 #during finetuning, only train classification layer and freeze rest. usually done for a few epochs (at least 1, more depends on size of dataset)
        if epoch <= epochs_to_finetune and (args.epochs_pretrain > 0 or args.state_dict_dir_net != ''):
            for param in net.module._add_on.parameters():
                param.requires_grad = False
            for param in params_to_train:
                param.requires_grad = False
            for param in params_to_freeze:
                param.requires_grad = False
            for param in params_backbone:
                param.requires_grad = False
            finetune = True
        
        else:
            finetune=False
            if frozen:
                # unfreeze backbone
                if epoch>(args.freeze_epochs):
                    for param in net.module._add_on.parameters():
                        param.requires_grad = True
                    for param in params_to_freeze:
                        param.requires_grad = True
                    for param in params_to_train:
                        param.requires_grad = True
                    for param in params_backbone:
                        param.requires_grad = True   
                    frozen = False
                # freeze first layers of backbone, train rest
                else:
                    for param in params_to_freeze:
                        param.requires_grad = True #Can be set to False if you want to train fewer layers of backbone
                    for param in net.module._add_on.parameters():
                        param.requires_grad = True
                    for param in params_to_train:
                        param.requires_grad = True
                    for param in params_backbone:
                        param.requires_grad = False
        
        print("\n Epoch", epoch, "frozen:", frozen, flush=True)
        if (epoch==args.epochs or epoch%30==0) and args.epochs>1:
            # SET SMALL WEIGHTS TO ZERO
            with torch.no_grad():
                torch.set_printoptions(profile="full")
                net.module._classification.weight.copy_(torch.clamp(net.module._classification.weight.data - 0.001, min=0.)) 
                print("Classifier weights: ", net.module._classification.weight[net.module._classification.weight.nonzero(as_tuple=True)], (net.module._classification.weight[net.module._classification.weight.nonzero(as_tuple=True)]).shape, flush=True)
                if args.bias:
                    print("Classifier bias: ", net.module._classification.bias, flush=True)
                torch.set_printoptions(profile="default")

        train_info = train_pipnet(algorithm, net, trainloader, optimizer_net, optimizer_classifier, scheduler_net, scheduler_classifier, criterion, epoch, args.epochs, device, pretrain=False, finetune=finetune)
        lrs_net+=train_info['lrs_net']
        lrs_classifier+=train_info['lrs_class']
        # Evaluate model
        eval_info = eval_pipnet(net, testloader, epoch, device, log)
        log.log_values('log_epoch_overview', epoch, eval_info['top1_accuracy'], eval_info['top5_accuracy'], eval_info['almost_sim_nonzeros'], eval_info['local_size_all_classes'], eval_info['almost_nonzeros'], eval_info['num non-zero prototypes'], train_info['train_accuracy'], train_info['loss'])

        wandb.log({
            "Train Loss": train_info['loss'],
            "Train Accuracy": train_info['train_accuracy'],
            "Test Top1 Accuracy": eval_info['top1_accuracy'],
            "Test Top5 Accuracy": eval_info['top5_accuracy']
        })

        with torch.no_grad():
            # save learning rate in figure
            plt.clf()
            plt.plot(lrs_net)
            plt.savefig(os.path.join(args.log_dir,'lr_net.png'))
            plt.clf()
            plt.plot(lrs_classifier)
            plt.savefig(os.path.join(args.log_dir,'lr_class.png'))
    
            torch.save({'model_state_dict': net.state_dict(), 'optimizer_net_state_dict': optimizer_net.state_dict(), 'optimizer_classifier_state_dict': optimizer_classifier.state_dict()}, os.path.join(os.path.join(args.log_dir, 'checkpoints'), f'net_trained_{args.algorithm}{epoch}'))

        # if epoch % 10 == 0:
        #     net.eval()
        #     topks = visualize_topk(net, projectloader, len(classes), device, f'visualised_prototypes_topk{epoch}', args)
        #     net.train()

    net.eval()
    # testset_img0_path = test_projectloader.dataset.x[0]
    # test_path = os.path.split(os.path.split(testset_img0_path)[0])[0]
    # vis_pred(net, test_path, classes, device, args) 
    torch.save({'model_state_dict': net.state_dict(), 'optimizer_net_state_dict': optimizer_net.state_dict(), 'optimizer_classifier_state_dict': optimizer_classifier.state_dict()}, os.path.join(os.path.join(args.log_dir, 'checkpoints'), f'net_trained_{args.algorithm}'))

    tr_wga, test_wga = evaluate_model(net, args, device)
    wandb.log({"test WGA": test_wga})
    wandb.log({"train WGA": tr_wga})

    for i in range(net.module._classification.weight.shape[0]):
        print(net.module._classification.weight[i])

    print("Done!", flush=True)
    wandb.finish()

def evaluate_model(net, args, device):
    net = deepcopy(net)
    dataset_class = get_dataset_class(args.dataset)
    split_names = ['tr','te']
    datasets = [dataset_class(split, None) for split in split_names]
    final_eval_loaders = [DataLoader(
        dataset=dset,
        batch_size=max(128, args.batch_size * 2),
        num_workers=0)
        for dset in datasets
    ]
    from functools import partial
    # Add a predict function because evaluation script expects one
    net.predict = lambda *args, **kwargs: partial(net.forward, inference=True)(*args, **kwargs)[2]

    final_results = {split: eval_metrics(net, loader, device)
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

    return final_results['tr']['min_group']['accuracy'], final_results['te']['min_group']['accuracy']

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