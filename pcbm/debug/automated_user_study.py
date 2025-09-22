import os
import json
import argparse
import torch
from openai import OpenAI

TASK_DESCRIPTIONS = {
    "waterbirds": "distinguish between WATERBIRDS and LANDBIRDS.",
    "celeba": "distinguish between people with BLONDE HAIR and DARK HAIR.",
    "metashift": "distinguish between common animal categories such as CATS and DOGS."
}

def load_api_key():
    with open(os.path.join(os.path.expanduser("~"), ".openai_api_key"), "r") as f:
        return f.read().strip()

def load_full_concepts(concepts_txt_path):
    with open(concepts_txt_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def call_gpt(client, concept, class_name, task_description):
    system_prompt = f"""You are a helpful assistant that classifies visual concepts as either SPURIOUS or NOT SPURIOUS.

The classification task is: {task_description}

A concept is considered SPURIOUS if:
1. It is NOT a physical or anatomical attribute of the object itself.
2. It may correlate with the label due to dataset bias (e.g., background scenery or co-occurring objects), but is not causally related to the object's identity.

Respond only with SPURIOUS or NOT SPURIOUS and a brief justification."""

    user_prompt = f"Class: {class_name}\nConcept: {concept}"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def evaluate_and_save(concept_json_path, dataset, concepts_txt_path, output_path):
    with open(concept_json_path, "r") as f:
        data = json.load(f)
    spurious_dict = data[0]  # <-- spurious concepts

    task_description = TASK_DESCRIPTIONS.get(dataset.lower(), f"classify objects from the dataset '{dataset}'.")

    full_concepts = load_full_concepts(concepts_txt_path)
    concept_to_index = {concept: i for i, concept in enumerate(full_concepts)}
    flags = torch.zeros(len(full_concepts), dtype=torch.int)

    client = OpenAI(api_key=load_api_key())

    for cls, concepts in spurious_dict.items():
        for concept in concepts:
            print(f"Evaluating [{cls}] -> '{concept}'...")
            result = call_gpt(client, concept, cls, task_description)
            if result.lower().startswith("spurious"):
                idx = concept_to_index.get(concept)
                if idx is not None:
                    flags[idx] = 1

    torch.save(flags, output_path)
    print(f"\n✅ Saved spurious flags to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate spurious concepts using GPT and save flag tensor.")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the concept JSON file")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (e.g. waterbirds)")
    parser.add_argument("--concepts_txt", type=str, required=True, help="Path to the full concept list (.txt)")
    parser.add_argument("--output", type=str, default="spurious_flags.pt", help="Path to save output torch tensor")
    args = parser.parse_args()

    evaluate_and_save(args.json_path, args.dataset, args.concepts_txt, args.output)

if __name__ == "__main__":
    main()
