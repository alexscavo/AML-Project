"""import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ✅ Update your dataset path here
LOVEDA_ROOT = 'data/loveda/'

# ✅ Define only the relevant subfolders
SUBSETS = ['train/Urban/'] # 'train/Rural/'

CLASS_NAMES = [
    'ignored',     # 0
    'background',  # 1
    'building',    # 2
    'road',        # 3
    'water',       # 4
    'barren',      # 5
    'forest',      # 6
    'agriculture'  # 7
]

NUM_CLASSES = len(CLASS_NAMES)

def compute_class_distribution():
    class_counts = np.zeros(NUM_CLASSES, dtype=np.uint64)

    for subset in SUBSETS:
        label_dir = os.path.join(LOVEDA_ROOT, subset, 'masks_png')
        if not os.path.exists(label_dir):
            raise FileNotFoundError(f"Missing directory: {label_dir}")

        label_files = [f for f in os.listdir(label_dir) if f.endswith('.png')]

        for file in tqdm(label_files, desc=f"Processing {subset}"):
            label_path = os.path.join(label_dir, file)
            label = np.array(Image.open(label_path))

            # Count pixels per class
            for class_id in range(NUM_CLASSES):
                class_counts[class_id] += np.sum(label == class_id)

    return class_counts

def main():
    class_counts = compute_class_distribution()
    total_pixels = np.sum(class_counts)

    print("\n📊 Class Distribution (pixel count):")
    for i, count in enumerate(class_counts):
        print(f"{i} ({CLASS_NAMES[i]}): {count:,} pixels ({(count / total_pixels)*100:.2f}%)")

    print("\n⚖️ Class Weights:")
    
    normalized = True
    if normalized:
        # inv_freq = 1.0 / (class_counts[1:] + 1e-8)
        # weights = inv_freq / np.sum(inv_freq)
        freq = class_counts / (total_pixels-class_counts[0])
        weights = 1 / freq
        print("Mean: ", np.mean(weights[1:]))
        weights = weights / np.mean(weights[1:])
    else:
        weights = 1.0 / (class_counts + 1e-8)
    weights[0] = 0.0 # exclude class 0 (ignored)
    weights_tensor = torch.tensor(weights, dtype=torch.float32)

    for i, w in enumerate(weights):
        print(f"{i} ({CLASS_NAMES[i]}): weight = {w:.6f}")

    print(f"\nUse in PyTorch loss as:\nCrossEntropyLoss(weight=torch.tensor([{', '.join(f'{w:.6f}' for w in weights)}]))")

    involved = ""

    involved = " + ".join(el.split("/")[1] for el in SUBSETS)

    # 📈 Save high-resolution figure for publication
    plt.figure(figsize=(8, 4))  # Wider aspect ratio fits better in columns
    bars = plt.bar(CLASS_NAMES, class_counts)

    plt.ylabel('Pixel Count', fontsize=10)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Optional: add count labels above bars
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height // 1_000:.0f}k',  # e.g., 12k
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 0),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    # 🔽 Save to file
    out_path = f'../Plots/class_distribution_{involved.lower()}.pdf'
    plt.savefig(out_path, dpi=300, bbox_inches='tight', transparent=True)

    print(f"Saved plot to: {out_path}")

if __name__ == '__main__':
    main()
"""


import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ─────────────────── UPDATE THESE PATHS AS NEEDED ───────────────────
LOVEDA_ROOT = 'data/loveda/'

# Define each split name with its corresponding subfolder
# (adjust 'val/Rural/' to wherever your Rural masks actually live)
SPLITS = {
    'Urban': ['train/Urban/'],
    'Rural': ['train/Rural/']
}

CLASS_NAMES = [
    'ignored',     # 0
    'background',  # 1
    'building',    # 2
    'road',        # 3
    'water',       # 4
    'barren',      # 5
    'forest',      # 6
    'agriculture'  # 7
]
NUM_CLASSES = len(CLASS_NAMES)


def compute_class_counts_for_subsets(subset_list):
    """
    Given a list of relative subfolders (e.g. ['train/Urban/']), 
    return a length-N array of pixel counts per class over all those subfolders.
    """
    counts = np.zeros(NUM_CLASSES, dtype=np.uint64)

    for subset in subset_list:
        label_dir = os.path.join(LOVEDA_ROOT, subset, 'masks_png')
        if not os.path.exists(label_dir):
            raise FileNotFoundError(f"Missing directory: {label_dir}")

        label_files = [f for f in os.listdir(label_dir) if f.endswith('.png')]
        for fname in tqdm(label_files, desc=f"Processing {subset}"):
            path = os.path.join(label_dir, fname)
            lbl = np.array(Image.open(path))
            # accumulate pixel‐counts for each class
            for c in range(NUM_CLASSES):
                counts[c] += np.sum(lbl == c)

    return counts


def main():
    # 1) Compute per‐class counts for each split (Urban vs. Rural)
    all_counts = {}
    for split_name, subset_list in SPLITS.items():
        all_counts[split_name] = compute_class_counts_for_subsets(subset_list)

    # 2) (Optional) Print out raw numbers and percentages
    print("\n📊 Per‐class pixel counts:")
    total_all = sum(all_counts[split].sum() for split in all_counts)
    for split_name, counts in all_counts.items():
        total_pixels = counts.sum()
        print(f"\n→ {split_name} split (total pixels = {total_pixels:,}):")
        for i, cls in enumerate(CLASS_NAMES):
            pct = counts[i] / total_pixels * 100
            print(f"   {i} ({cls}): {counts[i]:,} pixels ({pct:.2f}%)")

    # 3) (Optional) Compute and print class‐weights (example using Urban only)
    urban_counts = all_counts['Urban']
    freq = urban_counts / (urban_counts.sum() - urban_counts[0])
    weights = 1.0 / freq
    weights = weights / np.mean(weights[1:])  # normalize by mean of non-ignored
    weights[0] = 0.0  # ignore class=0
    weights_tensor = torch.tensor(weights, dtype=torch.float32)

    print("\n⚖️ Example class‐weights (based on Urban only):")
    for i, w in enumerate(weights):
        print(f"   {i} ({CLASS_NAMES[i]}): weight = {w:.6f}")
    print(f"\nUse in PyTorch as:\n"
          f"  CrossEntropyLoss(weight=torch.tensor([{', '.join(f'{w:.6f}' for w in weights)}]))")

    # 4) Plotting: grouped bar chart for Urban vs. Rural
    urban_counts = all_counts['Urban']
    rural_counts = all_counts['Rural']

    x = np.arange(NUM_CLASSES)   # one position per class
    width = 0.35                 # width of each bar

    plt.figure(figsize=(8, 4))
    bars_urban = plt.bar(x - width/2, urban_counts, width,
                         label='Urban', color='tab:blue', alpha=0.8)
    bars_rural = plt.bar(x + width/2, rural_counts, width,
                         label='Rural', color='tab:orange', alpha=0.8)

    # Axis labels, ticks, legend
    plt.ylabel('Pixel Count', fontsize=10)
    plt.xticks(x, CLASS_NAMES, rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.legend(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Annotate counts above each bar (in thousands, e.g., '12k')
    def annotate_bars(bar_container):
        for bar in bar_container:
            h = bar.get_height()
            plt.annotate(
                f'{h // 1_000:.0f}k',
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 0),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8
            )
    annotate_bars(bars_urban)
    annotate_bars(bars_rural)

    plt.tight_layout()
    out_path = f'../Plots/class_distribution_urban_vs_rural.pdf'
    plt.savefig(out_path, dpi=300, bbox_inches='tight', transparent=True)
    print(f"\n✅ Saved comparison plot to: {out_path}")


if __name__ == '__main__':
    main()
