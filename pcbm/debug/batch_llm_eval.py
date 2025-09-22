import os
import json
import torch
from pathlib import Path
from openai import OpenAI
from automated_user_study import load_api_key, call_gpt, load_full_concepts, TASK_DESCRIPTIONS

def evaluate_llm_for_run(concept_json_path, full_concept_path, dataset, output_path):
    with open(concept_json_path, "r") as f:
        data = json.load(f)
    spurious_dict = data[0]

    task_description = TASK_DESCRIPTIONS.get(dataset.lower(), f"classify objects from the dataset '{dataset}'")
    full_concepts = load_full_concepts(full_concept_path)
    concept_to_index = {concept: i for i, concept in enumerate(full_concepts)}
    flags = torch.zeros(len(full_concepts), dtype=torch.int)
    client = OpenAI(api_key=load_api_key())

    for cls, concepts in spurious_dict.items():
        for concept in concepts:
            print(f"Evaluating [{dataset}] [{cls}] → '{concept}'...")
            result = call_gpt(client, concept, cls, task_description)
            if result.lower().startswith("spurious"):
                idx = concept_to_index.get(concept)
                if idx is not None:
                    flags[idx] = 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(flags, output_path)
    print(f"✅ Saved to {output_path}")

def main():
    datasets = ["Waterbirds", "MetaShift", "CelebA"]
    runs = [1, 2, 3]

    for run in runs:
        seed = ((run - 1) % 3) + 1

        for dataset in datasets:
            concept_json_path = Path(f"concepts/{dataset}_{seed}_concepts.json")
            concepts_txt_path = Path(f"concepts/{dataset}.txt")
            output_path = Path(f"user_study/llm_run_{run}/pcbm_pruneweight_{dataset.lower()}_{seed}.pth")

            if not concept_json_path.exists():
                print(f"❌ Skipping missing: {concept_json_path}")
                continue

            evaluate_llm_for_run(
                concept_json_path=concept_json_path,
                full_concept_path=concepts_txt_path,
                dataset=dataset,
                output_path=output_path
            )

if __name__ == "__main__":
    main()
