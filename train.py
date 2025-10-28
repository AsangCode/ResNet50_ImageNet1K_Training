import os
import math
import torch
import pkg_resources

# Check PyTorch version
MINIMUM_TORCH_VERSION = '2.0.0'
current_torch_version = pkg_resources.get_distribution('torch').version
if pkg_resources.parse_version(current_torch_version) < pkg_resources.parse_version(MINIMUM_TORCH_VERSION):
    raise RuntimeError(f'PyTorch version should be >={MINIMUM_TORCH_VERSION}, but found {current_torch_version}')

# Print device information
print(f"\n[INFO] Using device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

# Enable CUDA optimizations and memory management
if torch.cuda.is_available():
    # Enable auto-tuner
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    # Memory management optimizations
    torch.cuda.empty_cache()  # Clear any allocated memory
    os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'  # Avoid memory fragmentation
    
    # Print GPU info
    gpu = torch.cuda.get_device_properties(0)
    print(f"\n[INFO] Using GPU: {gpu.name} with {gpu.total_memory/1024**3:.1f}GB memory")
    print(f"[INFO] CUDA Version: {torch.version.cuda}")
    print(f"[INFO] PyTorch CUDA: {torch.version.cuda}")
    if hasattr(torch.backends, 'cudnn'):
        print(f"[INFO] cuDNN Version: {torch.backends.cudnn.version()}")
import csv
import json
import argparse
import warnings
from datetime import datetime
from torch.amp import autocast, GradScaler
from lr_finder import LRFinder
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Suppress all warnings
warnings.filterwarnings('ignore')

# Suppress PyTorch warnings
if hasattr(torch.backends, 'cudnn'):
    torch.backends.cudnn.benchmark = True

# Disable PyTorch logging
os.environ["TORCH_LOGS"] = "0"
os.environ["TORCH_CUDNN_WARN"] = "0"

from model import create_model
from dataset import get_data_loaders

def save_epoch_metrics(metrics, filepath, rank):
    """Save metrics to CSV file (only on rank 0)"""
    if rank != 0:
        return
        
    file_exists = os.path.isfile(filepath)
    with open(filepath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)

def compute_accuracy(output, target, topk=(1, 5)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    device,
    writer,
    checkpoint_dir,
    logs_dir,
    rank,
    world_size
):
    # Create gradient scaler for AMP
    scaler = GradScaler()
    best_top1_acc = 0.0
    
    # Create log files (only on rank 0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_file = os.path.join(logs_dir, f'training_metrics_{timestamp}.csv')
    
    if rank == 0:
        training_config = {
            'start_time': timestamp,
            'num_epochs': num_epochs,
            'optimizer': optimizer.__class__.__name__,
            'scheduler': scheduler.__class__.__name__,
            'batch_size': train_loader.batch_size if hasattr(train_loader, 'batch_size') else None,
            'device': str(device),
            'world_size': world_size
        }
        with open(os.path.join(logs_dir, f'training_config_{timestamp}.json'), 'w') as f:
            json.dump(training_config, f, indent=4)
    
    for epoch in range(num_epochs):
        if rank == 0:
            print(f'Epoch {epoch+1}/{num_epochs}')
            print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = torch.zeros(1).to(device)
        running_corrects = torch.zeros(1).to(device)
        running_corrects_top5 = torch.zeros(1).to(device)  # Initialize top5 tensor
        total_samples = torch.zeros(1).to(device)
        
        train_iter = enumerate(train_loader)
        if rank == 0:
            train_iter = tqdm(train_iter, total=len(train_loader), desc=f'Training Epoch {epoch+1}')
        
        for i, (inputs, labels) in train_iter:
            # Handle last batch for WebDataset
            if isinstance(inputs, (list, tuple)):
                inputs = torch.stack(inputs)
            if isinstance(labels, (list, tuple)):
                labels = torch.stack(labels)
                
            # Move data to GPU asynchronously
            inputs = inputs.to(device, memory_format=torch.channels_last, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Zero gradients (accumulate_grad_batches=1 for better GPU utilization)
            optimizer.zero_grad(set_to_none=True)  # Slightly faster than zero_grad()
            
            # Forward pass with mixed precision
            with autocast(device_type='cuda', dtype=torch.float16):  # Use float16 for faster computation
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()
            
            # Update metrics
            running_loss += loss.item() * inputs.size(0)
            acc1, acc5 = compute_accuracy(outputs, labels, topk=(1, 5))
            running_top1 = acc1.item() * inputs.size(0) / 100.0  # Convert back to correct count
            running_top5 = acc5.item() * inputs.size(0) / 100.0
            running_corrects += running_top1
            running_corrects_top5 += running_top5  # Changed from = to +=
            total_samples += inputs.size(0)
            
            if rank == 0 and isinstance(train_iter, tqdm):
                train_iter.set_postfix({
                    'loss': f"{loss.item():.3f}",
                    'top1': f"{acc1.item():.2f}%",
                    'top5': f"{acc5.item():.2f}%"
                })
        
        # Gather metrics from all processes
        dist.all_reduce(running_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(running_corrects, op=dist.ReduceOp.SUM)
        dist.all_reduce(running_corrects_top5, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
        
        epoch_loss = running_loss.item() / total_samples.item()
        epoch_top1 = (running_corrects.item() / total_samples.item()) * 100
        epoch_top5 = (running_corrects_top5.item() / total_samples.item()) * 100
        
        if rank == 0:
            print(f'Train Loss: {epoch_loss:.4f} Top-1: {epoch_top1:.2f}% Top-5: {epoch_top5:.2f}%')
            writer.add_scalar('Loss/train', epoch_loss, epoch)
            writer.add_scalar('Accuracy/train_top1', epoch_top1, epoch)
            writer.add_scalar('Accuracy/train_top5', epoch_top5, epoch)
        
        # Validation phase
        model.eval()
        running_loss = torch.zeros(1).to(device)
        running_corrects = torch.zeros(1).to(device)
        running_corrects_top5 = torch.zeros(1).to(device)
        total_samples = torch.zeros(1).to(device)
        
        val_iter = enumerate(val_loader)
        if rank == 0:
            val_iter = tqdm(val_iter, total=len(val_loader), desc='Validation')
        
        with torch.no_grad(), autocast(device_type='cuda'):
            for i, (inputs, labels) in val_iter:
                if isinstance(inputs, (list, tuple)):
                    inputs = torch.stack(inputs)
                if isinstance(labels, (list, tuple)):
                    labels = torch.stack(labels)
                    
                inputs = inputs.to(device, memory_format=torch.channels_last)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                acc1, acc5 = compute_accuracy(outputs, labels, topk=(1, 5))
                running_top1 = acc1.item() * inputs.size(0) / 100.0
                running_top5 = acc5.item() * inputs.size(0) / 100.0
                running_corrects += running_top1
                running_corrects_top5 += running_top5
                total_samples += inputs.size(0)
                
                if rank == 0 and isinstance(val_iter, tqdm):
                    val_iter.set_postfix({
                        'loss': f"{loss.item():.3f}",
                        'top1': f"{acc1.item():.2f}%",
                        'top5': f"{acc5.item():.2f}%"
                    })
        
        # Gather validation metrics
        dist.all_reduce(running_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(running_corrects, op=dist.ReduceOp.SUM)
        dist.all_reduce(running_corrects_top5, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
        
        val_loss = running_loss.item() / total_samples.item()
        val_top1 = (running_corrects.item() / total_samples.item()) * 100
        val_top5 = (running_corrects_top5.item() / total_samples.item()) * 100
        
        if rank == 0:
            print(f'Val Loss: {val_loss:.4f} Top-1: {val_top1:.2f}% Top-5: {val_top5:.2f}%')
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/val_top1', val_top1, epoch)
            writer.add_scalar('Accuracy/val_top5', val_top5, epoch)
            
            # Save metrics to CSV
            metrics = {
                'epoch': epoch + 1,
                'train_loss': epoch_loss,
                'train_top1': epoch_top1,
                'train_top5': epoch_top5,
                'val_loss': val_loss,
                'val_top1': val_top1,
                'val_top5': val_top5,
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            save_epoch_metrics(metrics, metrics_file, rank)
            
            # Save checkpoint if best model (based on top-1 accuracy)
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_top1_acc': best_top1_acc,
                'train_loss': epoch_loss,
                'train_top1': epoch_top1,
                'train_top5': epoch_top5,
                'val_loss': val_loss,
                'val_top1': val_top1,
                'val_top5': val_top5,
                'world_size': world_size
            }
            
            # Save best model and periodic checkpoints
            if val_top1 > best_top1_acc:
                best_top1_acc = val_top1
                torch.save(checkpoint, os.path.join(checkpoint_dir, f'best_model_{timestamp}.pth'))
                print(f"\nNew best model saved! Top-1 Accuracy: {val_top1:.2f}%")
            
            # Save checkpoint every 5 epochs or on the last epoch
            if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
                torch.save(checkpoint, os.path.join(checkpoint_dir, f'epoch_{epoch+1}_model_{timestamp}.pth'))
                print(f"\nSaved checkpoint at epoch {epoch + 1}")
            
            # Early stopping if top-1 accuracy reaches 75%
            if val_top1 >= 75.0:
                print(f"\n🎉 Reached target accuracy of 75%! Current Top-1: {val_top1:.2f}%")
                print("Saving final model and stopping training...")
                torch.save(checkpoint, os.path.join(checkpoint_dir, f'target_reached_model_{timestamp}.pth'))
                return model
    
    return model

def setup(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Set device ID explicitly to avoid warnings
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
        device_id=rank  # Explicitly set device ID
    )

def cleanup():
    """Cleanup distributed training"""
    dist.destroy_process_group()

def train(rank, world_size, config):
    """Main training function for each process"""
    setup(rank, world_size)
    
    # Create directories (only on rank 0)
    if rank == 0:
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.runs_dir, exist_ok=True)
        os.makedirs(config.logs_dir, exist_ok=True)
    
    # Ensure all processes have directories created
    dist.barrier()
    
    # Set device
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    # Create data loaders with proper distributed setup
    # Optimize data loading for maximum GPU utilization
    train_loader, val_loader = get_data_loaders(
        batch_size=config.batch_size,
        num_workers=min(12, os.cpu_count() * 2),  # More workers for faster data loading
        distributed=True,
        world_size=world_size,
        rank=rank
    )
    
    # Enable non-blocking data transfers
    torch.cuda.set_device(device)
    torch.cuda.set_stream(torch.cuda.Stream())
    
    # Create model (training from scratch)
    model = create_model(
        distributed=True,
        local_rank=rank,
        channels_last=True,
        gradient_checkpointing=True
    )
    
    # Skip model compilation to avoid TF32 conflicts
    criterion = nn.CrossEntropyLoss()
    
    # Initialize optimizer with a low learning rate for LR finder
    optimizer = optim.SGD(
        model.parameters(),
        lr=1e-3,  # Initial learning rate for LR finder
        momentum=0.9,
        weight_decay=config.weight_decay
    )
    
    # Run LR finder only on rank 0
    if rank == 0:
        print("Running learning rate finder...")
        # Clear GPU memory before LR finder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        lr_finder = LRFinder(model, optimizer, criterion, device)
        # Reduce iterations and batch size for LR finder to save memory
        lr_finder.range_test(
            train_loader,
            end_lr=1,
            num_iter=200,
            smooth_f=0.05
        )
        
        min_loss_lr = lr_finder.history['lr'][lr_finder.history['loss'].index(min(lr_finder.history['loss']))]
        suggested_lr = min_loss_lr / 10
        
        print(f"\nLR Finder Results:")
        print(f"Minimum loss achieved at learning rate: {min_loss_lr:.6f}")
        print(f"Suggested learning rate (min_loss_lr/10): {suggested_lr:.6f}")
        
        lr_finder.plot()
        lr_finder.reset()
        
        init_lr = suggested_lr
    else:
        init_lr = None
    
    # Broadcast the initial learning rate to all processes
    if world_size > 1:
        init_lr = torch.tensor(init_lr if init_lr is not None else 0.0).to(device)
        dist.broadcast(init_lr, src=0)
        init_lr = init_lr.item()
    
    # Recreate optimizer with found learning rate
    optimizer = optim.SGD(
        model.parameters(),
        lr=init_lr,
        momentum=0.9,
        nesterov=True,  # Enable Nesterov momentum for better convergence
        weight_decay=config.weight_decay
    )
    
    # Learning rate schedule with OneCycleLR
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=init_lr * 1.5,  # Peak LR will be 1.5x the found optimal LR
        epochs=config.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.2,
        div_factor=10,
        final_div_factor=1e3
    )
    
    # TensorBoard writer (only on rank 0)
    writer = None
    if rank == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        writer = SummaryWriter(f'{config.runs_dir}/resnet50_training_{timestamp}')
    
    # Train the model
    try:
        model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=config.epochs,
            device=device,
            writer=writer,
            checkpoint_dir=config.checkpoint_dir,
            logs_dir=config.logs_dir,
            rank=rank,
            world_size=world_size
        )
    finally:
        if writer is not None:
            writer.close()
        cleanup()

class TrainingConfig:
    def __init__(self):
        # Optimize batch size for T4 GPU
        self.batch_size = 128   # Reduced batch size to prevent OOM during LR finder
        self.epochs = 100
        self.learning_rate = 0.2
        self.weight_decay = 1e-4
        self.num_workers = min(12, os.cpu_count() * 2)  # Dynamic optimization based on available CPU cores
        self.checkpoint_dir = 'checkpoints'
        self.runs_dir = 'runs'
        self.logs_dir = 'logs'
        
        # Memory optimization flags
        self.gradient_checkpointing = True  # Enable gradient checkpointing
        self.channels_last = True  # Enable channels last memory format
        self.empty_cache = True  # Empty CUDA cache between iterations

def get_user_input():
    config = TrainingConfig()
    
    print("\nCurrent training configuration:")
    print(f"1. Batch size per GPU: {config.batch_size}")
    print(f"2. Number of epochs: {config.epochs}")
    print(f"3. Initial learning rate: {config.learning_rate}")
    print(f"4. Weight decay: {config.weight_decay}")
    print(f"5. Number of workers per GPU: {config.num_workers}")
    print(f"6. Checkpoint directory: {config.checkpoint_dir}")
    print(f"7. TensorBoard runs directory: {config.runs_dir}")
    print(f"8. Logs directory: {config.logs_dir}")
    
    print("\nWould you like to modify any of these settings? (yes/no)")
    if input().lower().startswith('y'):
        print("\nEnter new values (press Enter to keep current value):")
        
        new_batch_size = input(f"Batch size per GPU [{config.batch_size}]: ")
        if new_batch_size:
            config.batch_size = int(new_batch_size)
            
        new_epochs = input(f"Number of epochs [{config.epochs}]: ")
        if new_epochs:
            config.epochs = int(new_epochs)
            
        new_weight_decay = input(f"Weight decay [{config.weight_decay}]: ")
        if new_weight_decay:
            config.weight_decay = float(new_weight_decay)
            
        new_num_workers = input(f"Number of workers per GPU [{config.num_workers}]: ")
        if new_num_workers:
            config.num_workers = int(new_num_workers)
            
        new_checkpoint_dir = input(f"Checkpoint directory [{config.checkpoint_dir}]: ")
        if new_checkpoint_dir:
            config.checkpoint_dir = new_checkpoint_dir
            
        new_runs_dir = input(f"TensorBoard runs directory [{config.runs_dir}]: ")
        if new_runs_dir:
            config.runs_dir = new_runs_dir
            
        new_logs_dir = input(f"Logs directory [{config.logs_dir}]: ")
        if new_logs_dir:
            config.logs_dir = new_logs_dir
    
    print("\nFinal configuration:")
    print(f"Batch size per GPU: {config.batch_size}")
    print(f"Number of epochs: {config.epochs}")
    print(f"Weight decay: {config.weight_decay}")
    print(f"Number of workers per GPU: {config.num_workers}")
    print(f"Checkpoint directory: {config.checkpoint_dir}")
    print(f"TensorBoard runs directory: {config.runs_dir}")
    print(f"Logs directory: {config.logs_dir}")
    
    print("\nProceed with training? (yes/no)")
    if not input().lower().startswith('y'):
        print("Training cancelled.")
        exit()
    
    return config

def main():
    # Get configuration from user
    config = get_user_input()
    
    # Get world size from environment variable or default to number of GPUs
    world_size = int(os.environ.get('WORLD_SIZE', torch.cuda.device_count()))
    
    if world_size > 1:
        mp.spawn(
            train,
            args=(world_size, config),
            nprocs=world_size,
            join=True
        )
    else:
        train(0, 1, config)

if __name__ == '__main__':
    main()