import json
import logging
import os
import signal
import sys
import time
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset_paths import TRAIN_DIR, VAL_DIR
from model import ResNet50

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cleanup_stale_processes():
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass
    os.system('fuser -k 29500/tcp')


def signal_handler(sig, frame):
    print("Cleaning up...")
    cleanup_stale_processes()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


torch.backends.cudnn.benchmark = True


class Params:
    def __init__(self):
        self.batch_size = 256
        self.name = "resnet_50_onecycle_distributed"
        self.workers = 12
        self.max_lr = 0.175
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.epochs = 65
        self.pct_start = 0.3
        self.div_factor = 25.0
        self.final_div_factor = 1e4

    def __repr__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class MetricLogger:
    def __init__(self, log_dir, rank):
        self.log_dir = log_dir
        self.rank = rank
        if rank == 0:
            os.makedirs(log_dir, exist_ok=True)
        self.metrics = []

    def log_metrics(self, epoch_metrics):
        if self.rank == 0:
            self.metrics.append(epoch_metrics)
            with open(os.path.join(self.log_dir, 'training_log.json'), 'w') as f:
                json.dump(self.metrics, f, indent=4)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def train_epoch(rank, dataloader, model, criterion, optimizer, scheduler, epoch, writer, scaler, metric_logger):
    model.train()
    running_loss = 0.0
    correct = 0
    correct_top5 = 0
    total = 0
    start_time = time.time()

    iterator = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}") if rank == 0 else enumerate(dataloader)

    for batch_idx, (inputs, targets) in iterator:
        inputs = inputs.cuda(rank, non_blocking=True)
        targets = targets.cuda(rank, non_blocking=True)

        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        scheduler.step()

        running_loss += loss.item()
        total += targets.size(0)

        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == targets).sum().item()

        _, pred_top5 = outputs.topk(5, 1, largest=True, sorted=True)
        correct_top5 += pred_top5.eq(targets.view(-1, 1).expand_as(pred_top5)).sum().item()

        if rank == 0 and batch_idx % 100 == 0:
            current_loss = loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            current_acc = 100 * correct / total
            current_acc5 = 100 * correct_top5 / total

            if isinstance(iterator, tqdm):
                iterator.set_postfix({
                    "loss": f"{current_loss:.4f}",
                    "acc": f"{current_acc:.2f}%",
                    "acc5": f"{current_acc5:.2f}%",
                    "lr": f"{current_lr:.6f}"
                })

            if writer is not None:
                step = epoch * len(dataloader.dataset) + (batch_idx + 1) * dataloader.batch_size
                writer.add_scalar('training loss', current_loss, step)
                writer.add_scalar('training accuracy', current_acc, step)
                writer.add_scalar('training top5 accuracy', current_acc5, step)
                writer.add_scalar('learning rate', current_lr, step)

    world_size = dist.get_world_size()
    tensors = {
        'loss': torch.tensor([running_loss], device=rank),
        'correct': torch.tensor([correct], device=rank),
        'correct_top5': torch.tensor([correct_top5], device=rank),
        'total': torch.tensor([total], device=rank)
    }

    for tensor in tensors.values():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    avg_loss = tensors['loss'].item() / (len(dataloader) * world_size)
    accuracy = 100 * tensors['correct'].item() / tensors['total'].item()
    accuracy_top5 = 100 * tensors['correct_top5'].item() / tensors['total'].item()
    epoch_time = time.time() - start_time

    if rank == 0:
        metrics = {
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'train_accuracy': accuracy,
            'train_accuracy_top5': accuracy_top5,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'epoch_time': epoch_time
        }
        metric_logger.log_metrics(metrics)
        logger.info(
            f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%, Top-5 Acc: {accuracy_top5:.2f}%, Time: {epoch_time:.2f}s"
        )


def validate(rank, dataloader, model, criterion, epoch, writer, train_loader, metric_logger, calc_acc5=True):
    model.eval()
    test_loss = 0.0
    correct = 0
    correct_top5 = 0
    total = 0

    iterator = tqdm(dataloader, desc=f"Validation Epoch {epoch}") if rank == 0 else dataloader

    with torch.no_grad():
        with autocast():
            for inputs, targets in iterator:
                inputs = inputs.cuda(rank, non_blocking=True)
                targets = targets.cuda(rank, non_blocking=True)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                total += targets.size(0)

                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == targets).sum().item()

                if calc_acc5:
                    _, pred_top5 = outputs.topk(5, 1, largest=True, sorted=True)
                    correct_top5 += pred_top5.eq(targets.view(-1, 1).expand_as(pred_top5)).sum().item()

    tensors = {
        'loss': torch.tensor([test_loss], device=rank),
        'correct': torch.tensor([correct], device=rank),
        'correct_top5': torch.tensor([correct_top5], device=rank),
        'total': torch.tensor([total], device=rank)
    }

    for tensor in tensors.values():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    if rank == 0:
        test_loss = tensors['loss'].item() / (len(dataloader) * dist.get_world_size())
        accuracy = 100 * tensors['correct'].item() / tensors['total'].item()
        accuracy_top5 = (
            100 * tensors['correct_top5'].item() / tensors['total'].item() if calc_acc5 else None
        )

        metrics = {
            'epoch': epoch,
            'test_loss': test_loss,
            'test_accuracy': accuracy,
            'test_accuracy_top5': accuracy_top5
        }
        metric_logger.log_metrics(metrics)

        if writer is not None:
            step = epoch * len(train_loader.dataset)
            writer.add_scalar('test loss', test_loss, step)
            writer.add_scalar('test accuracy', accuracy, step)
            if calc_acc5:
                writer.add_scalar('test accuracy5', accuracy_top5, step)

        logger.info(
            f"Validation Epoch {epoch} - Loss: {test_loss:.4f}, Acc: {accuracy:.2f}%, Top-5 Acc: {accuracy_top5:.2f}%"
        )


def main_worker(rank, world_size, params):
    try:
        setup(rank, world_size)

        if rank == 0:
            os.makedirs(os.path.join('checkpoints', params.name), exist_ok=True)
            os.makedirs(os.path.join('logs', params.name), exist_ok=True)
            os.makedirs(os.path.join('runs', params.name), exist_ok=True)

        dist.barrier()

        log_dir = os.path.join('logs', params.name)
        metric_logger = MetricLogger(log_dir, rank)

        train_dir = TRAIN_DIR
        val_dir = VAL_DIR

        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.RandomHorizontalFlip(0.5),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=train_transforms)
        train_sampler = DistributedSampler(train_dataset)

        train_loader = DataLoader(
            train_dataset,
            batch_size=params.batch_size,
            sampler=train_sampler,
            num_workers=params.workers,
            pin_memory=True,
            persistent_workers=params.workers > 0
        )

        val_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=256, antialias=True),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        val_dataset = torchvision.datasets.ImageFolder(root=val_dir, transform=val_transforms)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)

        val_loader = DataLoader(
            val_dataset,
            batch_size=params.batch_size,
            sampler=val_sampler,
            num_workers=params.workers,
            pin_memory=True,
            persistent_workers=params.workers > 0
        )

        torch.cuda.set_device(rank)
        num_classes = len(train_dataset.classes)
        model = ResNet50(num_classes=num_classes).cuda(rank)
        model = DDP(model, device_ids=[rank])

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            model.parameters(),
            lr=params.max_lr / params.div_factor,
            momentum=params.momentum,
            weight_decay=params.weight_decay
        )

        scaler = GradScaler()

        steps_per_epoch = len(train_loader)
        total_steps = params.epochs * steps_per_epoch

        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=params.max_lr,
            total_steps=total_steps,
            pct_start=params.pct_start,
            div_factor=params.div_factor,
            final_div_factor=params.final_div_factor
        )

        start_epoch = 0
        checkpoint_path = os.path.join('checkpoints', params.name, 'checkpoint.pth')

        if os.path.exists(checkpoint_path):
            print(f"Rank {rank}: Resuming training from checkpoint")
            map_location = {f'cuda:{0}': f'cuda:{rank}'}
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
            model.module.load_state_dict(checkpoint['model'])
            start_epoch = checkpoint['epoch'] + 1
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            scaler.load_state_dict(checkpoint['scaler'])
            assert params == checkpoint['params']

        writer = SummaryWriter(os.path.join('runs', params.name)) if rank == 0 else None

        validate(rank, val_loader, model, criterion, epoch=0, writer=writer, train_loader=train_loader,
                 metric_logger=metric_logger, calc_acc5=True)

        if rank == 0:
            print("Starting training")

        for epoch in range(start_epoch, params.epochs):
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)

            if rank == 0:
                print(f"Epoch {epoch}")

            train_epoch(
                rank,
                train_loader,
                model,
                criterion,
                optimizer,
                scheduler,
                epoch,
                writer,
                scaler,
                metric_logger
            )

            if rank == 0:
                checkpoint = {
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'epoch': epoch,
                    'params': params
                }
                torch.save(checkpoint, os.path.join('checkpoints', params.name, f'model_{epoch}.pth'))
                torch.save(checkpoint, checkpoint_path)

            validate(
                rank,
                val_loader,
                model,
                criterion,
                epoch + 1,
                writer,
                train_loader=train_loader,
                metric_logger=metric_logger,
                calc_acc5=True
            )

    except Exception as error:
        print(f"Error in rank {rank}: {error}")
        raise
    finally:
        cleanup()


def main():
    cleanup_stale_processes()
    params = Params()
    world_size = torch.cuda.device_count()

    if world_size == 0:
        raise RuntimeError('No CUDA devices available for distributed training.')

    try:
        mp.spawn(
            main_worker,
            args=(world_size, params),
            nprocs=world_size,
            join=True
        )
    except Exception as error:
        print(f"Error in main process: {error}")
        cleanup_stale_processes()
        sys.exit(1)


if __name__ == '__main__':
    main()
