"""
Algorithms to make some edit based on user provided spuriosity information
"""
import torch
import numpy as np
from tqdm import tqdm
import os, sys

import glob
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import algorithm

ALGORITHMS = [
    "Retrain",
    "Augment",
    "PermutationWeighting",
]

def get_algorithm(config):
    """Return the algorithm with the given name."""
    if config.algorithm not in globals():
        raise NotImplementedError(f"Algorithm not found: {config.algorithm}")
    return globals()[config.algorithm](config.prune_weight_path)

class Algorithm():
    """
    General Class for Intervention/Retraining
    """

    def get_weight_matrix(self, IMLModel):
        return IMLModel.classifier.weight
    
    def remove(self, model_info):
        all_concepts = model_info['concepts']
        train_projs = model_info['trainprojs']
        test_projs = model_info['testprojs']

        all_concept_names = list(all_concepts.keys())
        train_projs_filtered = train_projs[:, self.prune_weight != 1]
        test_projs_filtered = test_projs[:, self.prune_weight != 1]
        for i, weight in enumerate(self.prune_weight):
            if weight == 1:
                del all_concepts[all_concept_names[i]]
        return {'concepts': all_concepts, 'trainprojs': train_projs_filtered, 'testprojs': test_projs_filtered}
    
    def get_concepts(self, IMLModel, concept_indices, dataloader):
        self.path_to_concepts = "../pcbm/concept_banks/synthetic_concepts_disc"
        concept_set = {}

        for index in tqdm(concept_indices):
            concept_img_paths = glob.glob(f"{self.path_to_concepts}/{IMLModel.names[index]}/positives/*.png")
            list_concepts = []
            
            for img_path in concept_img_paths:
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img)
                list_concepts.append(img)
            concept_set[index] = torch.stack(list_concepts).cuda()
        return concept_set

    def set_transform(self, transform):
        self.transform = transform

class Remove(Algorithm, algorithm.Remove):
    """
    Just remove undesired concepts
    """
    def initialize(self, IMLModel, dataloader):
        return

class Retrain(Algorithm, algorithm.Retrain):
    """
    Just retrain model after removing undesired concepts
    """
    def initialize(self, IMLModel, dataloader):
        return

class Augment(Algorithm, algorithm.Augment):
    """
    Retrain the model with augmentation based on the spurious concepts
    """
    def augment(self, x, y, indices):
        self.mixing_coeff = 0.75
        batch_size, c, h, w = x.size()
        mixed_x = x.clone()

        for i in range(batch_size):
            # Optionally select concept from a class that it doesn't belong to
            # selected_class_idx_for_image = np.random.choice([c for c in self.concept_list.keys() if c != y[i]])
            # With only two classes, select randomly to not reintroduce any new biases
            selected_class_idx_for_image = np.random.choice([c for c in self.concept_list.keys()])

            concept_list = self.concept_list[selected_class_idx_for_image]
            concept_weights = self.concept_weights[selected_class_idx_for_image]

            # Select a concept from that class weighted by linear layer weights
            selected_concept_images = concept_list[np.random.choice(list(concept_list.keys()), p=concept_weights / np.sum(concept_weights))]
            
            # Select an image from the concept bank and perform mixup
            image = selected_concept_images[np.random.randint(0, len(selected_concept_images))]
            mixed_x[i] = self.mixing_coeff*mixed_x[i] + (1-self.mixing_coeff)*image.to(mixed_x.device)

        return mixed_x

class PermutationWeighting(Algorithm, algorithm.PermutationWeighting):
    """
    Perform Permutation Weighting
    """
    def get_activations(self, IMLModel, dataloader):
        train_projs = dataloader["projs"]
        train_lbls = dataloader["lbls"]

        indices = torch.where(self.prune_weight != 0)[0]
        self.Y = np.expand_dims(train_lbls, axis=1)
        self.spurious_concept_activations = train_projs[:, indices]

class CBDebug(Augment, PermutationWeighting, Algorithm):
    """
    CBDebug: Combining Permutation Weighting and Augmentation
    """
    def initialize(self, IMLModel, dataloader):
        self.name = "CBDebug"
        Augment.initialize(self, IMLModel, dataloader)
        PermutationWeighting.initialize(self, IMLModel, dataloader)
        self.gamma = 2
        weights = np.array(self.sample_weights)
        self.aug_probs = (weights.max() - weights)
        self.aug_probs = self.aug_probs / self.aug_probs.max()
        self.aug_probs = self.aug_probs ** self.gamma

    def augment(self, x, y, indices):
        batch_size, c, h, w = x.size()
        mixed_x = x.clone()
        self.mixing_coeff = 0.75

        for i in range(batch_size):
            p = self.aug_probs[indices[i]]
            if np.random.rand() > p:
                continue
            selected_class_idx_for_image = np.random.choice([c for c in self.concept_list.keys()])

            concept_list = self.concept_list[selected_class_idx_for_image]
            concept_weights = self.concept_weights[selected_class_idx_for_image]

            selected_concept_images = concept_list[np.random.choice(list(concept_list.keys()), p=concept_weights / np.sum(concept_weights))]
            
            image = selected_concept_images[np.random.randint(0, len(selected_concept_images))]

            mixed_x[i] = self.mixing_coeff*mixed_x[i] + (1-self.mixing_coeff)*image.to(mixed_x.device)

        return mixed_x
