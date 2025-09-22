import argparse
import os
import pickle
import numpy as np
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score

from data_utils import get_dataset
from concepts import ConceptBank
from models import PosthocLinearCBM, get_model
from training_tools import load_or_compute_projections, compute_projections
from debug.algorithm import get_algorithm


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept-bank", required=True, type=str, help="Path to the concept bank")
    parser.add_argument("--out-dir", required=True, type=str, help="Output folder for model/run info.")
    parser.add_argument("--backbone-dir", required=True, type=str, help="Folder where backbone is saved.")
    parser.add_argument("--dataset", default="cub", type=str)
    parser.add_argument("--pcbm-path", type=str)
    parser.add_argument("--prune_weight_path", type=str)
    parser.add_argument("--algorithm", type=str)
    parser.add_argument("--backbone-name", default="resnet18_cub", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--alpha", default=0.99, type=float, help="Sparsity coefficient for elastic net.")
    parser.add_argument("--lam", default=1e-5, type=float, help="Regularization strength.")
    parser.add_argument("--lr", default=1e-3, type=float)
    return parser.parse_args()


def run_linear_probe(args, algorithm, train_data, test_data):
    train_features, train_labels = train_data
    test_features, test_labels = test_data
    
    # We converged to using SGDClassifier. 
    # It's fine to use other modules here, this seemed like the most pedagogical option.
    # We experimented with torch modules etc., and results are mostly parallel.
    classifier = SGDClassifier(random_state=args.seed, loss="log_loss",
                               alpha=args.lam, l1_ratio=args.alpha, verbose=0,
                               penalty="elasticnet", max_iter=10000, class_weight='balanced')
    
    if algorithm.name == "PermutationWeighting" or algorithm.name == "CBDebug":
        classifier.fit(train_features, train_labels, sample_weight=algorithm.sample_weights)
    else:
        classifier.fit(train_features, train_labels)

    train_predictions = classifier.predict(train_features)
    train_accuracy = np.mean((train_labels == train_predictions).astype(float)) * 100.
    predictions = classifier.predict(test_features)
    test_accuracy = np.mean((test_labels == predictions).astype(float)) * 100.

    # Compute class-level accuracies. Can later be used to understand what classes are lacking some concepts.
    cls_acc = {"train": {}, "test": {}}
    for lbl in np.unique(train_labels):
        test_lbl_mask = test_labels == lbl
        train_lbl_mask = train_labels == lbl
        cls_acc["test"][lbl] = np.mean((test_labels[test_lbl_mask] == predictions[test_lbl_mask]).astype(float))
        cls_acc["train"][lbl] = np.mean(
            (train_labels[train_lbl_mask] == train_predictions[train_lbl_mask]).astype(float))
        print(f"{lbl}: {cls_acc['test'][lbl]}")

    run_info = {"train_acc": train_accuracy, "test_acc": test_accuracy,
                "cls_acc": cls_acc,
                }

    # If it's a binary task, we compute auc
    if test_labels.max() == 1:
        run_info["test_auc"] = roc_auc_score(test_labels, classifier.decision_function(test_features))
        run_info["train_auc"] = roc_auc_score(train_labels, classifier.decision_function(train_features))
        coef = torch.vstack([torch.tensor(-classifier.coef_), torch.tensor(classifier.coef_)])
        intercept = torch.vstack([torch.tensor(-classifier.intercept_), torch.tensor(classifier.intercept_)]).squeeze()
    else:
        coef = classifier.coef_
        intercept = classifier.intercept_
    return run_info, coef, intercept


def main(args, backbone, preprocess):
    all_concepts = pickle.load(open(args.concept_bank, 'rb'))
    all_concept_names = list(all_concepts.keys())
    print(f"Bank path: {args.concept_bank}. {len(all_concept_names)} concepts will be used.")
    concept_bank = ConceptBank(all_concepts, args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    train_loader, test_loader, idx_to_class, classes = get_dataset(args, preprocess)
    
    # Get a clean conceptbank string
    # e.g. if the path is /../../cub_resnet-cub_0.1_100.pkl, then the conceptbank string is resnet-cub_0.1_100
    # which means a bank learned with 100 samples per concept with C=0.1 regularization parameter for the SVM. 
    # See `learn_concepts_dataset.py` for details.
    conceptbank_source = args.concept_bank.split("/")[-1].split(".")[0] 
    num_classes = len(classes)
    
    posthoc_layer = torch.load(args.pcbm_path).float()
    posthoc_layer = posthoc_layer.to(args.device)

    # Unshuffled to ensure permutation weights are calculated on correct indices
    # And doesn't matter because we are passing to SGDClassifier
    train_embs, train_projs, train_lbls, test_embs, test_projs, test_lbls = load_or_compute_projections(args, backbone, posthoc_layer, train_loader, test_loader, attempt_load=False)

    algorithm = get_algorithm(args)
    algorithm.set_transform(preprocess)
    algorithm.initialize(posthoc_layer, {"projs": train_projs, "lbls": train_lbls})

    if algorithm.name == "Remove":
        # If only performing removal, just zero out concepts.
        for i, weight in enumerate(algorithm.prune_weight):
            if weight == 1:
                torch.nn.init.zeros_(posthoc_layer.classifier.weight[:, i])
    else:
        # If retraining, fully remove concepts from concept bank and precomputed projections.
        model_info = algorithm.remove({'concepts': all_concepts, 'trainprojs': train_projs, 'testprojs': test_projs})
        all_concepts, train_projs, test_projs = model_info['concepts'], model_info['trainprojs'], model_info['testprojs']
        all_concept_names = list(all_concepts.keys())
        print(f"After removal: {args.concept_bank}. {len(all_concept_names)} concepts will be used.")

        concept_bank = ConceptBank(all_concepts, args.device)
        posthoc_layer = PosthocLinearCBM(concept_bank, backbone_name=args.backbone_name, idx_to_class=idx_to_class, n_classes=num_classes)
        posthoc_layer = posthoc_layer.to(args.device)

        # Recompute projections to make sure they are correct.
        if algorithm.name == "CBDebug" or algorithm.name == "Augment":
            _, train_projs, train_lbls, _, test_projs, test_lbls = compute_projections(args, algorithm, backbone, posthoc_layer, train_loader, test_loader)

        run_info, weights, bias = run_linear_probe(args, algorithm, (train_projs, train_lbls), (test_projs, test_lbls))
        
        # Convert from the SGDClassifier module to PCBM module.
        posthoc_layer.set_weights(weights=weights, bias=bias)

    # Sorry for the model path hack. Probably i'll change this later.
    model_path = os.path.join(args.out_dir,
                              f"pcbm_{args.algorithm}_{args.dataset}__{args.backbone_name}__{conceptbank_source}__lam:{args.lam}__alpha:{args.alpha}__seed:{args.seed}.ckpt")
    torch.save(posthoc_layer, model_path)

    if num_classes > 1:
        # Prints the Top-5 Concept Weights for each class.
        print(posthoc_layer.analyze_classifier(k=5))

    print(f"Model saved to : {model_path}")
    # print(run_info)

if __name__ == "__main__":
    args = config()
    # Get the backbone from the model zoo.
    backbone, preprocess = get_model(args, backbone_name=args.backbone_name)
    backbone = backbone.to(args.device)
    backbone.eval()
    main(args, backbone, preprocess)
