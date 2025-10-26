import copy
import math
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

class LRFinder:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        self.history = {"lr": [], "loss": []}
        
        # Save the original state to restore after finding LR
        self.model_state = copy.deepcopy(model.state_dict())
        self.optimizer_state = copy.deepcopy(optimizer.state_dict())
    
    def range_test(self, train_loader, end_lr=10, num_iter=100, smooth_f=0.05, diverge_th=5):
        """
        Performs the learning rate range test.
        """
        # Reset test results
        self.history = {"lr": [], "loss": []}
        
        # Move model to device
        self.model.to(self.device)
        
        # Calculate LR multiplier
        lr_multiplier = (end_lr / self.optimizer.param_groups[0]['lr']) ** (1 / num_iter)
        
        # Initialize average loss
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        
        # Start training loop
        train_iter = iter(train_loader)
        pbar = tqdm(range(num_iter), desc="Finding optimal learning rate")
        for iteration in pbar:
            try:
                inputs, labels = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                inputs, labels = next(train_iter)
            
            # Move data to device
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update LR
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])
            
            # Compute smoothed loss
            batch_num += 1
            avg_loss = smooth_f * loss.item() + (1 - smooth_f) * avg_loss if batch_num > 1 else loss.item()
            smooth_loss = avg_loss / (1 - smooth_f ** batch_num)
            self.history["loss"].append(smooth_loss)
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{smooth_loss:.3f}", "lr": f"{self.optimizer.param_groups[0]['lr']:.3e}"})
            
            # Check if loss is exploding
            if batch_num > 1 and smooth_loss > diverge_th * best_loss:
                print("Loss is diverging, stopping early...")
                break
            
            # Save best loss
            if smooth_loss < best_loss or batch_num == 1:
                best_loss = smooth_loss
            
            # Update learning rate
            for param_group in self.optimizer.param_groups:
                param_group["lr"] *= lr_multiplier
    
    def plot(self, skip_start=10, skip_end=5):
        """
        Plots the learning rate range test results.
        """
        lrs = self.history["lr"]
        losses = self.history["loss"]
        
        # Skip first few and last few points for clearer visualization
        if skip_start:
            lrs = lrs[skip_start:]
            losses = losses[skip_start:]
        if skip_end:
            lrs = lrs[:-skip_end]
            losses = losses[:-skip_end]
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(lrs, losses)
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Finder')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.show()
    
    def reset(self):
        """
        Restores the model and optimizer to their initial states.
        """
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)
