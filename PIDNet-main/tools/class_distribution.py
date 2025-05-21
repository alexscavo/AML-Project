import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
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

    # Optional: plot a bar chart
    plt.figure(figsize=(10, 5))
    plt.bar(CLASS_NAMES, class_counts)
    plt.title(f'LoveDA Class Distribution ({involved})')
    plt.ylabel('Number of Pixels')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
"""

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

# ——— CONFIG ———
LOVEDA_ROOT    = 'data/loveda/'          # path to LoveDA root
SUBSETS        = ['train/Urban/']        # list of subfolders to include
CLASS_NAMES    = [
    'ignored', 'background', 'building',
    'road', 'water', 'barren',
    'forest', 'agriculture'
]
NUM_CLASSES    = len(CLASS_NAMES)
WEIGHT_METHOD  = 'inv'   # 'inv' for inverse-frequency, 'median' for median-frequency
# ——————————

def compute_class_distribution():
    counts = np.zeros(NUM_CLASSES, dtype=np.uint64)
    for subset in SUBSETS:
        mask_dir = os.path.join(LOVEDA_ROOT, subset, 'masks_png')
        if not os.path.isdir(mask_dir):
            raise FileNotFoundError(f"Missing directory: {mask_dir}")
        for fn in tqdm(os.listdir(mask_dir), desc=f"Scanning {subset}"):
            if not fn.endswith('.png'):
                continue
            arr = np.array(Image.open(os.path.join(mask_dir, fn)))
            # tally every class id (0–7)
            for cid in range(NUM_CLASSES):
                counts[cid] += np.sum(arr == cid)
    return counts

def compute_weights(counts):
    # drop class 0 (ignored) from all computations
    valid_counts = counts[1:]
    total_valid = valid_counts.sum()
    C = len(valid_counts)  # number of real classes

    # inverse-frequency: w_j = N / (C * n_j)
    inv_w = total_valid / (C * valid_counts)

    # median-frequency: w_j = median(f) / f_j,  f_j = n_j / N
    freqs    = valid_counts / total_valid
    median_f = np.median(freqs)
    med_w    = median_f / freqs

    # build full-length arrays (with zero at idx=0)
    w_inv_full   = np.concatenate(([0.], inv_w))
    w_med_full   = np.concatenate(([0.], med_w))

    return {
        'inv':    w_inv_full.astype(np.float32),
        'median': w_med_full.astype(np.float32),
    }


def main():
    counts = compute_class_distribution()
    total  = counts.sum()

    # report pixel distribution
    print("\n📊 Class Distribution:")
    for i, c in enumerate(counts):
        pct = c / total * 100
        print(f"  {i:>2} {CLASS_NAMES[i]:<12} : {c:,} px ({pct:.2f}%)")
    print(f"  — total pixels = {total:,}\n")

    # compute both sets of weights
    weights_dict = compute_weights(counts)
    weights = weights_dict[WEIGHT_METHOD]

    # print chosen weights
    print(f"⚖️  Using **{WEIGHT_METHOD}**-frequency weighting:")
    for i, w in enumerate(weights):
        print(f"  {i:>2} {CLASS_NAMES[i]:<12} : weight = {w:.6f}")
    print()

    # build PyTorch loss
    w_tensor = torch.tensor(weights, dtype=torch.float32)
    criterion = torch.nn.CrossEntropyLoss(weight=w_tensor, ignore_index=0)
    print("Use in PyTorch:")
    print("  criterion = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=0)")

    # optional: plot class distribution
    plt.figure(figsize=(8,4))
    plt.bar(CLASS_NAMES, counts)
    plt.title(f"LoveDA Class Counts ({', '.join(SUBSETS)})")
    plt.ylabel('Pixels')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
"""