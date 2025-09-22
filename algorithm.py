import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold

class Algorithm():
    """
    General Class for Removal/Retraining
    """
    def __init__(self, prune_weight_path) -> None:
        # Load vector marking concepts as spurious (1) or not (0) 
        self.prune_weight = torch.load(prune_weight_path)

class Remove(Algorithm):
    """
    Just remove concepts
    """
    def __init__(self, prune_weight_path) -> None:
        self.name = "Remove"
        super().__init__(prune_weight_path)

class Retrain(Algorithm):
    """
    Just retrain model after removing undesired concepts
    """
    def __init__(self, prune_weight_path) -> None:
        self.name = "Retrain"
        super().__init__(prune_weight_path)

class Augment(Algorithm):
    """
    Take the spurious concepts and perform augmentation to break correlations
    """
    def __init__(self, prune_weight_path) -> None:
        self.name = "Augment"
        super().__init__(prune_weight_path)

    def initialize(self, IMLModel, dataloader):
        classification_weight = self.get_weight_matrix(IMLModel)

        spurious_concept_weights = classification_weight.data * self.prune_weight.cuda() # Do this before pruning
        all_spurious_concepts = torch.nonzero(spurious_concept_weights)[:, 1].tolist()
        unique_spurious = []
        for x in all_spurious_concepts:
            if x not in unique_spurious:
                unique_spurious.append(x)
        concepts = self.get_concepts(IMLModel, unique_spurious, dataloader)

        concept_list = {}
        concept_weights = {}
        for c in range(classification_weight.shape[0]):
            # For each class store spurious concepts and weights
            concept_indices = torch.nonzero(spurious_concept_weights[c, :]).squeeze().tolist()
            if not isinstance(concept_indices, list):
                concept_indices = [concept_indices]
 
            if len(concept_indices) > 0:
                # concept_list[c] = concepts
                concept_list[c] = {c:concepts[c] for c in concept_indices}
                concept_weights[c] = [torch.abs(spurious_concept_weights)[c, idx].cpu().item() for idx in concept_indices]

        self.concept_list = concept_list
        self.concept_weights = concept_weights

    def augment(self, x, y):
        raise NotImplementedError


class ProtoPDebug(Algorithm):
    """
    https://github.com/abonte/protopdebug
    Take desired interventions from the user for which concepts are spurious
    If confounded add cutout to F
    """
    def __init__(self, prune_weight_path) -> None:
        self.name = "ProtoPDebug"
        super().__init__(prune_weight_path)

    def initialize(self, IMLModel, dataloader):
        classification_weight = self.get_weight_matrix(IMLModel)

        spurious_concept_weights = classification_weight.data * self.prune_weight.cuda() # Do this before pruning
        self.forbid_concepts = {}
        for c in range(classification_weight.shape[0]):
            forbid_concept_indices = torch.nonzero(spurious_concept_weights[c, :]).flatten()
            concepts = self.get_concepts(IMLModel, forbid_concept_indices.tolist(), dataloader)
            if len(concepts) > 0:
                self.forbid_concepts[c] = concepts

    def compute_loss(self, IMLModel):
        return self.forbidding_loss(IMLModel, self.forbid_concepts)

    def forbidding_loss(self, IMLModel, all_forbid_concepts):
        max_activations = []
        for class_idx in all_forbid_concepts.keys():
            forbid_concepts = self.extract_concepts(all_forbid_concepts[class_idx])
            # Compute max over all activated concepts and all forbid concepts
            max_activations.append(torch.max(self.forward(IMLModel, forbid_concepts)))

        return torch.mean(torch.stack(max_activations))

    def extract_concepts(self, cls_concepts):
        # Creates a list that has each of the concepts and randomly chooses one of the images to represent said concept
        return torch.stack([cls_concepts[c][np.random.randint(0, len(cls_concepts[c]))] for c in cls_concepts.keys()]).cuda()

class PermutationWeighting(Algorithm):
    """
    Permutation Weighting with Causally Motivated Shortcut Removal
    """
    def __init__(self, prune_weight_path) -> None:
        self.name = "PermutationWeighting"
        self.num_permutations = 5
        self.num_splits = 5
        super().__init__(prune_weight_path)

    def initialize(self, IMLModel, dataloader):
        self.get_activations(IMLModel, dataloader)
        n = len(self.Y)
        C = np.concatenate([np.zeros(n), np.ones(n)])
        u_i = np.zeros(len(C))
        
        clf = GradientBoostingClassifier(n_estimators=200)
        kf = KFold(n_splits=self.num_splits, shuffle=True)

        for _ in tqdm(range(self.num_permutations)):
            permuted_indices = np.random.permutation(n)
            X_combined = np.vstack([np.hstack([self.Y, self.spurious_concept_activations]), np.hstack([self.Y[permuted_indices], self.spurious_concept_activations])])

            for train_idx, val_idx in kf.split(X_combined):
                X_train, X_val = X_combined[train_idx], X_combined[val_idx]
                C_train, C_val = C[train_idx], C[val_idx]

                clf.fit(X_train, C_train)
                probs = clf.predict_proba(X_val)[:, 1]
                u_i[val_idx] += probs / (1 - probs)
        
        self.sample_weights = u_i[:n] / self.num_permutations
