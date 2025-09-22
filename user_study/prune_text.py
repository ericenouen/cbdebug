import torch
import tkinter as tk
import argparse
import os
import json

TASK_DESCRIPTIONS = {
    "waterbirds": "Task: Classify between waterbirds and landbirds.",
    "metashift": "Task: Classify between cats and dogs.",
}

MAX_PER_PAGE = 18
GRID_COLUMNS = 5
class ConceptSelectorPagedGUI:
    def __init__(self, root, concepts_per_class, core_concepts_per_class, save_path, dataset_name, full_concepts):
        self.root = root
        self.concepts_per_class = concepts_per_class  # spurious concepts
        self.core_concepts_per_class = core_concepts_per_class
        self.class_names = list(set(concepts_per_class.keys()) | set(core_concepts_per_class.keys()))
        self.full_concepts = full_concepts
        self.save_path = save_path

        self.flags_per_class = {
            cls: [0] * len(concepts_per_class.get(cls, [])) for cls in self.class_names
        }

        self.root.title("Spurious Concept Selector")
        self.root.geometry("1600x1000")
        self.root.minsize(800, 500)

        task_desc = TASK_DESCRIPTIONS.get(dataset_name.lower(), f"Task: {dataset_name}")
        tk.Label(root, text=task_desc, font=("Helvetica", 14, "bold"),
                 wraplength=1600, justify="center").pack(pady=(10, 5))

        container = tk.Frame(root)
        container.pack(fill="both", expand=True, padx=20, pady=20)

        self.canvas = tk.Canvas(container, highlightthickness=0)
        scrollbar = tk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.root.bind_all("<MouseWheel>", self.on_mouse_wheel)

        self.button_refs = {}
        self.render_all_classes()

        self.save_btn = tk.Button(root, text="Save Responses", command=self.save_results,
                                  bg="green", fg="black", font=("Helvetica", 12))
        self.save_btn.pack(pady=10)

    def on_mouse_wheel(self, event):
        if event.delta > 0:
            self.canvas.yview_scroll(-1, "units")
        else:
            self.canvas.yview_scroll(1, "units")

    def render_all_classes(self):
        for cls in self.class_names:
            group_frame = tk.LabelFrame(self.scrollable_frame, text=cls, font=("Helvetica", 12, "bold"),
                                        padx=10, pady=10, labelanchor="n")
            group_frame.pack(fill="x", expand=True, pady=8, padx=15)

            # --- Core Concepts (visually inert) ---
            if self.core_concepts_per_class.get(cls):
                tk.Label(group_frame, text="Core Concepts", font=("Helvetica", 10, "italic")).pack()
                core_frame = tk.Frame(group_frame)
                core_frame.pack(anchor="w")

                for i, concept in enumerate(self.core_concepts_per_class[cls]):
                    lbl = tk.Label(core_frame, text=concept, width=25, height=2,
                                wraplength=180, justify="center",
                                bg="#b8f2b0", fg="black", relief="solid", borderwidth=1)
                    lbl.grid(row=i // GRID_COLUMNS, column=i % GRID_COLUMNS, padx=5, pady=5)
                    lbl.bind("<Enter>", lambda e, l=lbl: l.config(cursor="arrow"))


            # --- Spurious Concepts (clickable) ---
            if self.concepts_per_class.get(cls):
                tk.Label(group_frame, text="Spurious Candidates", font=("Helvetica", 10, "bold")).pack(pady=(10, 0))
                spurious_frame = tk.Frame(group_frame)
                spurious_frame.pack(anchor="w")

                self.button_refs[cls] = []
                for i, concept in enumerate(self.concepts_per_class[cls]):
                    btn = tk.Button(spurious_frame, text=concept, width=25, height=2,
                                    wraplength=180, justify="center",
                                    relief="raised", command=lambda c=cls, i=i: self.toggle(c, i))
                    btn.grid(row=i // GRID_COLUMNS, column=i % GRID_COLUMNS, padx=5, pady=5)
                    self.button_refs[cls].append(btn)
                    self.update_button_style(cls, i)

    def toggle(self, cls, index):
        self.flags_per_class[cls][index] ^= 1
        self.update_button_style(cls, index)

    def update_button_style(self, cls, index):
        flag = self.flags_per_class[cls][index]
        btn = self.button_refs[cls][index]
        if flag == 1:
            btn.config(relief="sunken", bg="#444", fg="red")
        else:
            btn.config(relief="raised", bg="SystemButtonFace", fg="black")

    def save_results(self):
        concept_to_index = {concept: i for i, concept in enumerate(self.full_concepts)}
        final_flags = torch.zeros(len(self.full_concepts), dtype=torch.int)

        for cls, concepts in self.concepts_per_class.items():
            flags = self.flags_per_class[cls]
            for i, concept in enumerate(concepts):
                if flags[i] == 1:
                    idx = concept_to_index.get(concept)
                    if idx is not None:
                        final_flags[idx] = 1

        torch.save(final_flags, self.save_path)
        self.root.quit()
        
def run_gui(dataset_name, seed, results_dir):
    # Load combined concept file
    with open(f"pcbm_models/{dataset_name}_{seed}_concepts.json", 'r') as f:
        concepts_json = json.load(f)

    spurious_concepts_per_class = concepts_json[0]
    core_concepts_per_class = concepts_json[1]

    with open(f"pcbm_models/{dataset_name}.txt", 'r') as f:
        full_concepts = [line.strip() for line in f if line.strip()]

    save_path = f"{results_dir}/pcbm_pruneweight_{dataset_name}_{seed}.pth"

    root = tk.Tk()
    root.state('zoomed')
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)
    root.focus_force()

    app = ConceptSelectorPagedGUI(
        root,
        spurious_concepts_per_class,
        core_concepts_per_class,
        save_path,
        dataset_name,
        full_concepts
    )
    root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select Spurious Concepts (Per-Class GUI)")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., waterbirds, celeba, etc.)')
    parser.add_argument('--seed', type=int, required=True, help='Path to save the .pth binary tensor')
    parser.add_argument('--results_dir', type=str, required=True, help='Path to shared results folder (e.g., user_study_results)')
    args = parser.parse_args()

    run_gui(args.dataset, args.seed, args.results_dir)
