import os
from torchvision import datasets, models, transforms
import torch

def get_transforms():
    data_transforms = {
        'train': transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.400, 0.445, 0.467], [0.2742, 0.259, 0.263])
        ]),
        'dev': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.400, 0.445, 0.467], [0.2742, 0.259, 0.263])
        ]),
    }

    return data_transforms

def get_dataloaders(data_dir):

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = get_transforms()

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),data_transforms[x]) for x in ['train', 'dev']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size= 8, shuffle=True, num_workers=4) for x in ['train', 'dev']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'dev']}
    class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names

