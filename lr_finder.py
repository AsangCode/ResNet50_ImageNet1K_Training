import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch_lr_finder import LRFinder

from dataset_paths import TRAIN_DIR
from model import ResNet50
from train import Params


def find_lr(start_lr=1e-7, end_lr=10, num_iter=100, output_dir='lr_finder_plots'):
    params = Params()
    print(f"Find LR with params: start_lr={start_lr}, end_lr={end_lr}, num_iter={num_iter}")

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f"Using {device} device")

    train_dir = TRAIN_DIR

    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.RandomHorizontalFlip(0.5),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=params.workers,
        pin_memory=True
    )

    model = ResNet50(num_classes=len(train_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=start_lr,
        momentum=params.momentum,
        weight_decay=params.weight_decay
    )

    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(train_loader, start_lr=start_lr, end_lr=end_lr, num_iter=num_iter, step_mode='exp')

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'lr_finder_{timestamp}_start{start_lr}_end{end_lr}_iter{num_iter}.png'
    filepath = os.path.join(output_dir, filename)

    fig, ax = plt.subplots()
    lr_finder.plot(ax=ax)
    plt.title(f'Learning Rate Finder (iter: {num_iter})')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Plot saved to: {filepath}")
    lr_finder.reset()


if __name__ == '__main__':
    import fire

    fire.Fire(find_lr)
