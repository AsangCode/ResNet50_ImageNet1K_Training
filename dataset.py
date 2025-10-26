import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.utils.data.distributed import DistributedSampler

# Get dataset directory from environment variable or use default
DATA_DIR = os.getenv('DATASET_PATH', '/mnt/imagenet/extracted')

def create_transforms(train=True):
    """Create transforms for training/validation"""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.2)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

def get_data_loaders(
    batch_size=128,
    num_workers=4,
    distributed=False,
    world_size=1,
    rank=0
):
    """Create train and validation data loaders using ImageFolder"""
    
    # Enable memory pinning and channels last format
    torch.backends.cudnn.benchmark = True
    
    # Create transforms
    train_transform = create_transforms(train=True)
    val_transform = create_transforms(train=False)
    
    # Create datasets using ImageFolder
    train_dir = os.path.join(DATA_DIR, 'train')
    val_dir = os.path.join(DATA_DIR, 'val')
    
    print(f"Loading training data from: {train_dir}")
    print(f"Loading validation data from: {val_dir}")
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    
    # Create samplers for distributed training
    train_sampler = None
    val_sampler = None
    
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),  # Shuffle only if not using sampler
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=False
    )
    
    return train_loader, val_loader
