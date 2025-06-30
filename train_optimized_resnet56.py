#!/usr/bin/env python3
"""
Optimized Training Script for ResNet56 with Advanced Pruning
Author: Manus AI
Date: June 30, 2025

This script provides a complete training pipeline for ResNet56 with optimized pruning
capabilities, including progressive pruning, knowledge distillation, and advanced
fine-tuning strategies.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import argparse
import os
import time
import logging
from typing import Dict, Tuple, Optional
import json
from tqdm import tqdm

# Import our optimized pruning implementation
from optimized_prune_resnet56 import (
    OptimizedPruner, 
    PruningConfig, 
    prune_resnet56_optimized
)

# Import ResNet56 model (assuming it's available)
import sys
sys.path.append('/home/ubuntu/implementation-of-pruning-filters')
from netModels.ResNet56 import MyResNet56

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KnowledgeDistillationLoss(nn.Module):
    """Knowledge distillation loss for pruned model training."""
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.3):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_outputs: torch.Tensor, teacher_outputs: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        """
        Compute combined knowledge distillation and classification loss.
        
        Args:
            student_outputs: Outputs from pruned (student) model
            teacher_outputs: Outputs from original (teacher) model  
            targets: Ground truth labels
        """
        
        # Standard classification loss
        ce_loss = self.ce_loss(student_outputs, targets)
        
        # Knowledge distillation loss
        student_soft = F.log_softmax(student_outputs / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_outputs / self.temperature, dim=1)
        kd_loss = self.kl_loss(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # Combined loss
        total_loss = (1 - self.alpha) * ce_loss + self.alpha * kd_loss
        
        return total_loss


class OptimizedTrainer:
    """Advanced trainer for ResNet56 with optimized pruning."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize data loaders
        self.train_loader, self.test_loader = self._prepare_data()
        
        # Initialize model
        self.model = self._prepare_model()
        self.original_model = None  # Will store teacher model for distillation
        
        # Initialize pruning configuration
        self.pruning_config = self._create_pruning_config()
        
        # Training state
        self.current_epoch = 0
        self.best_accuracy = 0.0
        self.training_history = []
        
    def _prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        """Prepare CIFAR-10 data loaders with appropriate transforms."""
        
        # Data transforms
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # Datasets
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test
        )
        
        # Data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.args.batch_size, 
            shuffle=True, num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.args.batch_size, 
            shuffle=False, num_workers=4, pin_memory=True
        )
        
        logger.info(f"Data prepared: {len(train_dataset)} training, {len(test_dataset)} test samples")
        
        return train_loader, test_loader
    
    def _prepare_model(self) -> nn.Module:
        """Initialize ResNet56 model."""
        
        model = MyResNet56()
        model = model.to(self.device)
        
        # Load pretrained weights if specified
        if self.args.pretrained_path and os.path.exists(self.args.pretrained_path):
            logger.info(f"Loading pretrained weights from {self.args.pretrained_path}")
            checkpoint = torch.load(self.args.pretrained_path, map_location=self.device)
            model.load_state_dict(checkpoint)
        
        logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
        
        return model
    
    def _create_pruning_config(self) -> PruningConfig:
        """Create pruning configuration from command line arguments."""
        
        config = PruningConfig(
            stage1_ratio=self.args.stage1_ratio,
            stage2_ratio=self.args.stage2_ratio,
            stage3_ratio=self.args.stage3_ratio,
            phase1_epochs=self.args.phase1_epochs,
            phase2_epochs=self.args.phase2_epochs,
            phase3_epochs=self.args.phase3_epochs,
            phase1_lr=self.args.phase1_lr,
            phase2_lr=self.args.phase2_lr,
            phase3_lr=self.args.phase3_lr,
            distillation_weight=self.args.distillation_weight,
            temperature=self.args.temperature
        )
        
        return config
    
    def train_epoch(self, model: nn.Module, optimizer: optim.Optimizer, 
                   criterion: nn.Module, epoch: int, 
                   teacher_model: Optional[nn.Module] = None) -> Dict:
        """Train model for one epoch."""
        
        model.train()
        if teacher_model is not None:
            teacher_model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            if teacher_model is not None and isinstance(criterion, KnowledgeDistillationLoss):
                with torch.no_grad():
                    teacher_outputs = teacher_model(inputs)
                loss = criterion(outputs, teacher_outputs, targets)
            else:
                loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.3f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_stats = {
            'epoch': epoch,
            'train_loss': running_loss / len(self.train_loader),
            'train_accuracy': 100. * correct / total
        }
        
        return epoch_stats
    
    def evaluate(self, model: nn.Module, criterion: nn.Module) -> Dict:
        """Evaluate model on test set."""
        
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.test_loader, desc='Evaluating'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct / total
        avg_loss = test_loss / len(self.test_loader)
        
        eval_stats = {
            'test_loss': avg_loss,
            'test_accuracy': accuracy
        }
        
        logger.info(f"Test Results - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        return eval_stats
    
    def fine_tune_phase(self, model: nn.Module, epochs: int, lr: float, 
                       teacher_model: Optional[nn.Module] = None) -> List[Dict]:
        """Fine-tune model for specified number of epochs."""
        
        # Setup optimizer and criterion
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        if teacher_model is not None:
            criterion = KnowledgeDistillationLoss(
                temperature=self.pruning_config.temperature,
                alpha=self.pruning_config.distillation_weight
            )
        else:
            criterion = nn.CrossEntropyLoss()
        
        phase_history = []
        
        for epoch in range(epochs):
            # Training
            train_stats = self.train_epoch(model, optimizer, criterion, epoch, teacher_model)
            
            # Evaluation
            eval_stats = self.evaluate(model, nn.CrossEntropyLoss())
            
            # Combine statistics
            epoch_stats = {**train_stats, **eval_stats}
            phase_history.append(epoch_stats)
            
            # Update learning rate
            scheduler.step()
            
            # Save best model
            if eval_stats['test_accuracy'] > self.best_accuracy:
                self.best_accuracy = eval_stats['test_accuracy']
                self.save_checkpoint(model, epoch_stats, is_best=True)
            
            logger.info(f"Epoch {epoch}: Train Acc: {train_stats['train_accuracy']:.2f}%, "
                       f"Test Acc: {eval_stats['test_accuracy']:.2f}%")
        
        return phase_history
    
    def execute_optimized_training(self):
        """Execute the complete optimized training pipeline."""
        
        logger.info("Starting optimized ResNet56 training with progressive pruning")
        
        # Phase 1: Initial training (if needed)
        if not self.args.skip_initial_training:
            logger.info("Phase 1: Initial training")
            initial_history = self.fine_tune_phase(
                self.model, 
                epochs=self.args.initial_epochs,
                lr=self.args.initial_lr
            )
            self.training_history.extend(initial_history)
        
        # Store original model for knowledge distillation
        self.original_model = torch.nn.DataParallel(self.model).cuda() if torch.cuda.device_count() > 1 else self.model
        self.original_model = self.original_model.to(self.device)
        
        # Phase 2: Progressive pruning
        if self.args.enable_pruning:
            logger.info("Phase 2: Progressive pruning")
            
            # Create pruner
            pruner = OptimizedPruner(self.model, self.pruning_config)
            
            # Execute progressive pruning with fine-tuning
            for phase in range(3):
                phase_num = phase + 1
                logger.info(f"Pruning Phase {phase_num}")
                
                # Determine phase parameters
                if phase_num == 1:
                    phase_ratio = self.pruning_config.phase1_ratio
                    fine_tune_epochs = self.pruning_config.phase1_epochs
                    fine_tune_lr = self.pruning_config.phase1_lr
                elif phase_num == 2:
                    phase_ratio = self.pruning_config.phase2_ratio
                    fine_tune_epochs = self.pruning_config.phase2_epochs
                    fine_tune_lr = self.pruning_config.phase2_lr
                else:
                    phase_ratio = self.pruning_config.phase3_ratio
                    fine_tune_epochs = self.pruning_config.phase3_epochs
                    fine_tune_lr = self.pruning_config.phase3_lr
                
                # Execute pruning phase
                phase_stats = pruner.progressive_prune_phase(phase_ratio)
                
                # Fine-tune after pruning
                logger.info(f"Fine-tuning after pruning phase {phase_num}")
                fine_tune_history = self.fine_tune_phase(
                    self.model,
                    epochs=fine_tune_epochs,
                    lr=fine_tune_lr,
                    teacher_model=self.original_model if self.args.use_distillation else None
                )
                
                # Store results
                phase_stats['fine_tune_history'] = fine_tune_history
                self.training_history.append(phase_stats)
                
                # Update model reference
                self.model = pruner.model
        
        # Phase 3: Final evaluation and statistics
        logger.info("Phase 3: Final evaluation")
        final_stats = self.evaluate(self.model, nn.CrossEntropyLoss())
        
        # Generate comprehensive statistics
        if hasattr(self, 'original_model') and self.args.enable_pruning:
            pruning_stats = pruner.get_model_statistics()
            final_stats.update(pruning_stats)
        
        # Save final results
        self.save_final_results(final_stats)
        
        logger.info("Optimized training completed successfully!")
        logger.info(f"Final test accuracy: {final_stats['test_accuracy']:.2f}%")
        
        if 'compression_ratio' in final_stats:
            logger.info(f"Compression ratio: {final_stats['compression_ratio']:.2%}")
    
    def save_checkpoint(self, model: nn.Module, stats: Dict, is_best: bool = False):
        """Save model checkpoint."""
        
        os.makedirs(self.args.checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'stats': stats,
            'args': vars(self.args),
            'pruning_config': self.pruning_config.__dict__
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.args.checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.args.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with accuracy: {stats['test_accuracy']:.2f}%")
    
    def save_final_results(self, final_stats: Dict):
        """Save final training results and statistics."""
        
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        # Save comprehensive results
        results = {
            'final_statistics': final_stats,
            'training_history': self.training_history,
            'arguments': vars(self.args),
            'pruning_configuration': self.pruning_config.__dict__
        }
        
        results_path = os.path.join(self.args.output_dir, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save final model
        final_model_path = os.path.join(self.args.output_dir, 'final_model.pth')
        torch.save(self.model.state_dict(), final_model_path)
        
        logger.info(f"Final results saved to {self.args.output_dir}")


def parse_arguments():
    """Parse command line arguments."""
    
    parser = argparse.ArgumentParser(description='Optimized ResNet56 Training with Advanced Pruning')
    
    # Model and data arguments
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--pretrained-path', type=str, default='', help='Path to pretrained model weights')
    
    # Training arguments
    parser.add_argument('--initial-epochs', type=int, default=200, help='Initial training epochs')
    parser.add_argument('--initial-lr', type=float, default=0.1, help='Initial learning rate')
    parser.add_argument('--skip-initial-training', action='store_true', help='Skip initial training phase')
    
    # Pruning arguments
    parser.add_argument('--enable-pruning', action='store_true', help='Enable progressive pruning')
    parser.add_argument('--stage1-ratio', type=float, default=0.15, help='Stage 1 pruning ratio')
    parser.add_argument('--stage2-ratio', type=float, default=0.40, help='Stage 2 pruning ratio')
    parser.add_argument('--stage3-ratio', type=float, default=0.60, help='Stage 3 pruning ratio')
    
    # Progressive pruning schedule
    parser.add_argument('--phase1-epochs', type=int, default=10, help='Phase 1 fine-tuning epochs')
    parser.add_argument('--phase2-epochs', type=int, default=15, help='Phase 2 fine-tuning epochs')
    parser.add_argument('--phase3-epochs', type=int, default=20, help='Phase 3 fine-tuning epochs')
    parser.add_argument('--phase1-lr', type=float, default=0.01, help='Phase 1 learning rate')
    parser.add_argument('--phase2-lr', type=float, default=0.005, help='Phase 2 learning rate')
    parser.add_argument('--phase3-lr', type=float, default=0.001, help='Phase 3 learning rate')
    
    # Knowledge distillation arguments
    parser.add_argument('--use-distillation', action='store_true', help='Use knowledge distillation')
    parser.add_argument('--distillation-weight', type=float, default=0.3, help='Distillation loss weight')
    parser.add_argument('--temperature', type=float, default=4.0, help='Distillation temperature')
    
    # Output arguments
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--output-dir', type=str, default='./results', help='Output directory')
    
    return parser.parse_args()


def main():
    """Main training function."""
    
    # Parse arguments
    args = parse_arguments()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # Create trainer and execute training
    trainer = OptimizedTrainer(args)
    trainer.execute_optimized_training()


if __name__ == '__main__':
    main()

