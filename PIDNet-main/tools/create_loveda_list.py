import os

def create_lst_file(image_dir, label_dir, output_lst):
    # List and sort files numerically
    images = sorted(os.listdir(image_dir), key=lambda x: int(os.path.splitext(x)[0]))
    labels = sorted(os.listdir(label_dir), key=lambda x: int(os.path.splitext(x)[0]))
    
    os.makedirs(os.path.dirname(output_lst), exist_ok=True)  # Ensure the directory exists
    
    with open(output_lst, 'w') as f:
        for img, lbl in zip(images, labels):
            # Generate full paths and normalize to use forward slashes
            img_path = os.path.join(image_dir, img).replace("\\", "/")
            lbl_path = os.path.join(label_dir, lbl).replace("\\", "/")
            # Write formatted line with consistent spacing
            f.write(f"{img_path} {lbl_path}\n")

# Paths to the LoveDA dataset directories
train_image_dir = "data/loveda/train/Urban/images_png"
train_label_dir = "data/loveda/train/Urban/masks_png"
test_image_dir = "data/loveda/val/Rural/images_png"
test_label_dir = "data/loveda/val/Rural/masks_png"

train_lst_path = "data/list/loveda/train.lst"
test_lst_path = "data/list/loveda/val.lst"

# Create .lst files
create_lst_file(train_image_dir, train_label_dir, train_lst_path)
create_lst_file(test_image_dir, test_label_dir, test_lst_path)
