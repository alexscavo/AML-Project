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
urban_train_image_dir = "data/loveda/train/Urban/images_png"
urban_train_label_dir = "data/loveda/train/Urban/masks_png"

urban_test_image_dir = "data/loveda/val/Urban/images_png"
urban_test_label_dir = "data/loveda/val/Urban/masks_png"

rural_train_image_dir = "data/loveda/train/Rural/images_png"
rural_train_label_dir = "data/loveda/train/Rural/masks_png"

rural_test_image_dir = "data/loveda/val/Rural/images_png"
rural_test_label_dir = "data/loveda/val/Rural/masks_png"



# train on urban - test on urban
train_lst_path = "data/list/loveda/urban_urban/train.lst"
test_lst_path = "data/list/loveda/urban_urban/val.lst"

# Create .lst files
create_lst_file(urban_train_image_dir, urban_train_label_dir, train_lst_path)
create_lst_file(urban_test_image_dir, urban_test_label_dir, test_lst_path)

train_lst_path = "data/list/loveda/urban_rural/train.lst"
target_lst_path = "data/list/loveda/rural/train.lst"
test_lst_path = "data/list/loveda/urban_rural/val.lst"

# Create .lst files
create_lst_file(urban_train_image_dir, urban_train_label_dir, train_lst_path)
create_lst_file(rural_train_image_dir, rural_train_label_dir, target_lst_path)
create_lst_file(rural_test_image_dir, rural_test_label_dir, test_lst_path)
