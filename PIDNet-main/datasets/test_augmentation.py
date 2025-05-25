from loveda import LoveDA
import albumentations as A
import torch

if __name__ == '__main__':
    train_transform = A.Compose([A.ColorJitter(p=0.5)])
    train_dataset = LoveDA(root='data/',
                           list_path="list/loveda/urban_rural/train.lst",
                           num_classes=8,
                           ignore_label=0,
                           base_size=1024,
                           crop_size=(1024, 1024),
                           scale_factor=16,
                           transform=train_transform,
                           show_img=True)

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=6,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
        drop_last=True)
    
    for i_iter, batch in enumerate(trainloader, 0):
        img, lbl, _, _ , _=  batch

    