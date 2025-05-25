import os
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
