import requests
import os
import pickle
import torch
import clip
import argparse
import numpy as np
from tqdm import tqdm


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=str)
    parser.add_argument("--classes", default="cifar10", type=str)
    parser.add_argument("--backbone-name", default="clip:RN50", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    return parser.parse_args()


@torch.no_grad()
def learn_conceptbank(args, concept_list, scenario):
    concept_dict = {}
    for concept in tqdm(concept_list):
        # Note: You can try other forms of prompting, e.g. "photo of {concept}" etc. here.
        text = clip.tokenize(f"{concept}").to("cuda")
        text_features = model.encode_text(text).cpu().numpy()
        text_features = text_features/np.linalg.norm(text_features)
        # store concept vectors in a dictionary. Adding the additional terms to be consistent with the
        # `ConceptBank` class (see `concepts/concept_utils.py`).
        concept_dict[concept] = (text_features, None, None, 0, {})

    print(f"# concepts: {len(concept_dict)}")
    concept_dict_path = os.path.join(args.out_dir, f"multimodal_concept_{args.backbone_name}_{scenario}.pkl")
    pickle.dump(concept_dict, open(concept_dict_path, 'wb'))
    print(f"Dumped to : {concept_dict_path}")


if __name__ == "__main__":
    args = config()
    model, _ = clip.load(args.backbone_name.split(":")[1], device=args.device, download_root=args.out_dir)
    concept_cache = {}

    with open(f'concepts/{args.classes}.txt', 'r') as f:
        all_concepts = [line.strip() for line in f if line.strip()]
    learn_conceptbank(args, all_concepts, args.classes)
