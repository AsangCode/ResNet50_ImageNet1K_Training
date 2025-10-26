import torch
import torch.nn as nn
import torchvision.models as models
from torch.distributed import DistributedDataParallel as DDP
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F

class ResNet50(nn.Module):
    def __init__(self, num_classes=1000, channels_last=True, gradient_checkpointing=True):
        super(ResNet50, self).__init__()
        # Initialize ResNet50 from scratch without pretrained weights
        self.model = models.resnet50(weights=None)
        
        # Replace final FC layer for ImageNet-1K
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        # Enable channels last memory format for better GPU performance
        if channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        
        # Enable gradient checkpointing to save memory
        if gradient_checkpointing:
            self.model.train()
            for module in self.model.modules():
                if isinstance(module, nn.Sequential):
                    module.register_forward_hook(self._gradient_checkpointing_hook)
    
    def _gradient_checkpointing_hook(self, module, input, output):
        if module.training:
            output.requires_grad_(True)
            output.retain_grad()
    
    def forward(self, x):
        # Ensure input is in channels last format if enabled
        if x.is_contiguous(memory_format=torch.channels_last):
            return self.model(x)
        else:
            return self.model(x.contiguous(memory_format=torch.channels_last))

def create_model(distributed=True, local_rank=None, **kwargs):
    """Factory function to create and wrap model for distributed training"""
    model = ResNet50(**kwargs)
    
    if distributed:
        if local_rank is None:
            raise ValueError("local_rank must be provided for distributed training")
        model = model.to(local_rank)
        model = DDP(model, device_ids=[local_rank])
    
    return model