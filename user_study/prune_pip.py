import os
import re
import torch
import argparse
import logging
import tkinter as tk
from pathlib import Path
from PIL import Image, ImageTk

logging.basicConfig(level=logging.INFO)
class UserStudyGUI:
    def __init__(self, root, image_groups, results_path, dataset, seed):
        self.root = root
        self.image_groups = image_groups
        self.results_path = results_path
        self.dataset = dataset
        self.seed = seed

        self.prune_weight = torch.zeros(768)

        self.scroll_canvas = tk.Canvas(root)
        self.scroll_frame = tk.Frame(self.scroll_canvas, bg="#e0e0e0")
        self.scrollbar = tk.Scrollbar(root, orient="vertical", command=self.scroll_canvas.yview)

        self.scroll_canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scroll_canvas.pack(side=tk.LEFT, fill="both", expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill="y")

        # Enable mousewheel scrolling on the canvas - Global scroll binding
        self.root.bind_all("<MouseWheel>", self.on_mouse_wheel)

        self.scroll_canvas.create_window((0, 0), window=self.scroll_frame, anchor='nw')
        self.scroll_frame.bind("<Configure>", lambda e: self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all")))

        self.label_vars = []
        # Grid container (centered by grid layout)
        self.grid_frame = tk.Frame(self.scroll_frame, bg="#e0e0e0")
        self.grid_frame.pack(pady=20)

        num_cols = 2
        for idx, group in enumerate(self.image_groups):
            row = idx // num_cols
            col = idx % num_cols

            proto_idx = int(re.search(r'grid_topk_(\d+).png', group['topk'].name).group(1))
            var = tk.BooleanVar(value=False)
            self.label_vars.append((proto_idx, var))

            card = tk.Frame(self.grid_frame, bd=1, relief="solid", bg="#f7f7f7",
                            padx=10, pady=10, cursor="hand2")
            card.grid(row=row, column=col, padx=20, pady=20, sticky="n")

            def make_toggle(card_ref=card, var_ref=var):
                return lambda e: self.toggle_label(card_ref, var_ref)

            card.bind("<Button-1>", make_toggle())

            # Topk image
            topk_img = Image.open(group['topk']).resize((600, 60))
            topk_tk = ImageTk.PhotoImage(topk_img)
            topk_lbl = tk.Label(card, image=topk_tk, borderwidth=0, highlightthickness=0, bg="#f7f7f7")
            topk_lbl.image = topk_tk
            topk_lbl.pack()
            topk_lbl.bind("<Button-1>", make_toggle())

            # Activations
            # Hidden activation frame
            act_frame = tk.Frame(card, bg="#f7f7f7")
            act_frame.pack_forget()  # Hidden by default

            # Button to show activations
            def make_toggle_activations(frame_ref=act_frame, btn_text=None):
                def toggle():
                    if frame_ref.winfo_ismapped():
                        frame_ref.pack_forget()
                    else:
                        frame_ref.pack(pady=5)
                return toggle

            toggle_text = tk.StringVar(value="Show Example Images")
            toggle_btn = tk.Button(card, textvariable=toggle_text, command=make_toggle_activations(), bg="#b0b0b0", font=("Helvetica", 10))
            toggle_btn.pack(pady=(10, 0))
            for act_img_path in group['activations'][:3]:
                act_img = Image.open(act_img_path).resize((160, 160))
                act_tk = ImageTk.PhotoImage(act_img)
                lbl = tk.Label(act_frame, image=act_tk, bg="#f7f7f7")
                lbl.image = act_tk
                lbl.pack(side=tk.LEFT, padx=5)
                lbl.bind("<Button-1>", make_toggle())
            self.update_style(card, var.get())


        # Save button
        save_btn = tk.Button(root, text="Save and Quit", command=self.save_results,
                             font=("Helvetica", 14), height=5)
        save_btn.pack(fill="x", pady=10)

    def on_mouse_wheel(self, event):
        if event.delta > 0:  # Scrolling up
            self.scroll_canvas.yview_scroll(-1, "units")
        else:  # Scrolling down
            self.scroll_canvas.yview_scroll(1, "units")

    def toggle_label(self, card, var):
        var.set(not var.get())
        self.update_style(card, var.get())

    def update_style(self, card, is_spurious):
        if is_spurious:
            card.config(bg="black")
            for child in card.winfo_children():
                child.config(bg="black")
                if isinstance(child, (tk.Label, tk.Frame)):
                    try:
                        child.config(fg="white")
                    except:
                        pass
        else:
            card.config(bg="#f7f7f7")
            for child in card.winfo_children():
                child.config(bg="#f7f7f7")
                if isinstance(child, (tk.Label, tk.Frame)):
                    try:
                        child.config(fg="black")
                    except:
                        pass

    def save_results(self):
        for proto_idx, var in self.label_vars:
            self.prune_weight[proto_idx] = 1 if var.get() else 0
        self.results_path.mkdir(parents=True, exist_ok=True)
        filename = f"pip_pruneweight_{self.dataset}_{self.seed}.pth"
        save_path = self.results_path / filename
        torch.save(self.prune_weight, save_path)
        logging.info(f"Prune weights saved to {save_path}")
        self.root.quit()

def load_image_groups(base_dir):
    topk_dir = base_dir / "visualised_prototypes_topk"
    groups = []

    for topk_img in sorted(topk_dir.glob("grid_topk_*.png"), key=lambda x: x.name):
        match = re.search(r'grid_topk_(\d+).png', topk_img.name)
        if not match:
            continue
        proto_idx = match.group(1)
        prototype_dir = base_dir / f"visualised_prototypes/prototype_{proto_idx}"

        if prototype_dir.exists():
            activation_imgs = sorted(
                prototype_dir.glob("*.png"),
                key=lambda x: -float(re.search(r'p\d+_\d+_(\d+\.\d+)', x.name).group(1))
            )[:3]
        else:
            activation_imgs = []

        groups.append({
            'topk': topk_img,
            'activations': activation_imgs
        })

    return groups

def get_next_run_folder(base_dir):
    base_dir.mkdir(parents=True, exist_ok=True)
    existing_runs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
    next_run_id = 1
    if existing_runs:
        last_run = max(int(d.name.split("_")[1]) for d in existing_runs if d.name.split("_")[1].isdigit())
        next_run_id = last_run + 1
    next_run_dir = base_dir / f"run_{next_run_id}"
    next_run_dir.mkdir()
    return next_run_dir

def user_study(dataset, seed, results_path):
    # Set base path
    base_path = Path("pip_models") / f"{dataset}_{seed}"
    
    # Load images
    image_groups = load_image_groups(base_path)

    # Start GUI
    root = tk.Tk()
    root.state('zoomed')
    root.title("User Study - Label Spurious Features")
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)
    root.focus_force()

    # Initialize the GUI with necessary params
    app = UserStudyGUI(root, image_groups, results_path=results_path, dataset=dataset, seed=seed)
    root.mainloop()

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='User Study GUI')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., waterbirds, metashift)')
    parser.add_argument('--seed', type=int, required=True, help='Seed value for randomization')
    parser.add_argument('--results_dir', type=str, required=True, help='Path to shared results folder (e.g., user_study_results)')
    args = parser.parse_args()

    # Get next run folder (run_1, run_2, ...)
    results_path = Path(args.results_dir)

    # Run user study with the provided dataset and seed
    user_study(args.dataset, args.seed, results_path)