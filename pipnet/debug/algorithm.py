"""
Algorithms to make some edit based on user provided spuriosity information
"""
import torch
import numpy as np
from tqdm import tqdm
import os, sys
import torch.nn.functional as F

from util.vis_pipnet import get_concepts

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import algorithm

ALGORITHMS = [
    "Retrain",
    "Augment",
    "ProtoPDebug",
    "CBDebug",
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
    def reinitialize(self, IMLModel):
        weight_matrix = self.get_weight_matrix(IMLModel)
        # Randomly reinitialize unused concepts
        with torch.no_grad():
            for i in range(len(self.prune_weight)):
                if sum(weight_matrix[:, i]) == 0:
                    weight_matrix[:, i] = torch.normal(mean=1.0, std=0.01, size=weight_matrix[:, i].shape).cuda()
        return IMLModel

    def remove(self, IMLModel):
        weight_matrix = self.get_weight_matrix(IMLModel)
        for i, weight in enumerate(self.prune_weight):
            if weight == 1:
                torch.nn.init.zeros_(weight_matrix[:, i])
        return IMLModel
    
    def get_weight_matrix(self, IMLModel):
        return IMLModel.module._classification.weight
    
    def get_concepts(self, IMLModel, concept_indices, dataloader):
        return get_concepts(IMLModel, concept_indices, dataloader)
    
    def forward(self, IMLModel, concepts):
        _, concept_activations, _ = IMLModel(concepts, inference=True)
        return concept_activations

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
    Take the spurious concepts from concept bank and perform mixup to break correlations
    """
    def __init__(self, prune_weight_path):
        self.num_patches = 5
        super().__init__(prune_weight_path)
    
    def augment(self, x, y, indices):
        batch_size, c, h, w = x.size()
        mixed_x = x.clone()

        for i in range(batch_size):
            for _ in range(self.num_patches):
                selected_class_idx_for_image = np.random.choice([c for c in self.concept_list.keys()])

                concept_list = self.concept_list[selected_class_idx_for_image]
                concept_weights = self.concept_weights[selected_class_idx_for_image]

                # select a concept from that class based on weights of linear layer
                selected_concept_images = concept_list[np.random.choice(list(concept_list.keys()), p=concept_weights / np.sum(concept_weights))]
                
                # select random patch out of top-10 closest
                patch = selected_concept_images[np.random.randint(0, len(selected_concept_images))]
                patch = torch.squeeze(patch, dim=0)

                patch_h, patch_w = patch.shape[1], patch.shape[2]
                assert patch_h <= h and patch_w <= w, "Patch size must fit within the image dimensions."

                top_left_x = np.random.randint(0, w - patch_w)
                top_left_y = np.random.randint(0, h - patch_h)

                mixed_x[i, :, top_left_y:top_left_y + patch_h, top_left_x:top_left_x + patch_w] = patch

        return mixed_x

class ProtoPDebug(Algorithm, algorithm.ProtoPDebug):
    """
    https://github.com/abonte/protopdebug
    """

class PermutationWeighting(Algorithm, algorithm.PermutationWeighting):
    """
    Permutation Weighting with Causally Motivated Shortcut Removal
    """
    def get_activations(self, IMLModel, dataloader):
        self.criterion = torch.nn.NLLLoss(reduction='none')

        indices = torch.where(self.prune_weight != 0)[0]

        dataset = dataloader.dataset
        unshuf_dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
        IMLModel.eval()
        spurious_concept_activations = []
        labels = []
        with torch.no_grad():
            for batch, (x1, x2, y, _) in enumerate(tqdm(unshuf_dataloader)):
                activated_concepts = IMLModel(x1)[1]

                batch_spur_concepts_only = torch.stack([activated_concepts[:, idx] for idx in indices], dim=1)
                spurious_concept_activations.append(batch_spur_concepts_only.cpu())
                labels.append(y)

        self.Y = torch.cat(labels).unsqueeze(1).numpy()
        self.spurious_concept_activations = torch.cat(spurious_concept_activations).numpy()

    def compute_loss(self, out, normalization_multiplier, y, idx):
        sample_weights = torch.tensor(self.sample_weights[idx]).cuda()
        sample_weights = torch.cat([sample_weights, sample_weights])

        softmax_inputs = torch.log1p(out**normalization_multiplier)
        class_losses = self.criterion(F.log_softmax((softmax_inputs),dim=1),y.cuda())
    
        weighted_loss = torch.mean(sample_weights * class_losses)

        return weighted_loss

class CBDebug(Augment, PermutationWeighting, Algorithm):
    """
    CBDebug with Causally Motivated Shortcut Removal
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
        indices = torch.cat([indices, indices])

        for i in range(batch_size):
            p = self.aug_probs[indices[i]]
            if np.random.rand() > p:
                continue 
            for _ in range(self.num_patches):
                selected_class_idx_for_image = np.random.choice([c for c in self.concept_list.keys()])

                concept_list = self.concept_list[selected_class_idx_for_image]
                concept_weights = self.concept_weights[selected_class_idx_for_image]

                # select a concept from that class based on weights of linear layer
                selected_concept_images = concept_list[np.random.choice(list(concept_list.keys()), p=concept_weights / np.sum(concept_weights))]
                
                # select random patch out of top-10 closest
                patch = selected_concept_images[np.random.randint(0, len(selected_concept_images))]
                patch = torch.squeeze(patch, dim=0)

                patch_h, patch_w = patch.shape[1], patch.shape[2]
                assert patch_h <= h and patch_w <= w, "Patch size must fit within the image dimensions."

                top_left_x = np.random.randint(0, w - patch_w)
                top_left_y = np.random.randint(0, h - patch_h)

                mixed_x[i, :, top_left_y:top_left_y + patch_h, top_left_x:top_left_x + patch_w] = patch

        return mixed_x
