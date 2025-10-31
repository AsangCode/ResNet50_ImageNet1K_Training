import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.models as models


class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


def create_model(distributed=True, local_rank=None, num_classes=1000):
    model = ResNet50(num_classes=num_classes)

    if distributed:
        if local_rank is None:
            raise ValueError("local_rank must be provided for distributed training")
        model = model.cuda(local_rank)
        model = DDP(model, device_ids=[local_rank])

    return model