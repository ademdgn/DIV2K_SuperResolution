"""
Improved ESRGAN Training Script
Optimized for better performance and training stability
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import vgg19
from PIL import Image 
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import argparse
import time
import logging
from tqdm import tqdm
import json
import random
import math

# Import our models
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.models.esrgan import RDBNet, Discriminator, PerceptualLoss
    from src.utils.data_utils import DIV2KDataset
except ImportError:
    from models.esrgan import RDBNet, Discriminator, PerceptualLoss
    from utils.data_utils import DIV2KDataset

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImprovedDIV2KDataset(Dataset):
    """Improved DIV2K Dataset with better augmentation and preprocessing"""
    
    def __init__(self, hr_dir, lr_dir=None, hr_size=192, scale_factor=4, mode='train'):
        self.hr_dir = Path(hr_dir)
        self.lr_dir = Path(lr_dir) if lr_dir else None
        self.hr_size = hr_size
        self.lr_size = hr_size // scale_factor
        self.scale_factor = scale_factor
        self.mode = mode
        
        # Get file lists
        self.hr_files = sorted(list(self.hr_dir.glob("*.png")) + list(self.hr_dir.glob("*.jpg")))
        
        logger.info(f"Found {len(self.hr_files)} high resolution images")
        
        if self.mode == 'train':
            # Training augmentations
            self.transform = transforms.Compose([
                transforms.RandomCrop(self.hr_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.ToTensor(),
            ])
        else:
            # Validation transforms (no augmentation)
            self.transform = transforms.Compose([
                transforms.RandomCrop(self.hr_size),
                transforms.ToTensor(),
            ])
    
    def __len__(self):
        return len(self.hr_files) * (4 if self.mode == 'train' else 1)  # 4x augmentation for training
    
    def __getitem__(self, idx):
        real_idx = idx % len(self.hr_files)
        hr_path = self.hr_files[real_idx]
        
        # Load HR image
        hr_image = Image.open(hr_path).convert('RGB')
        
        # Ensure minimum size
        min_size = self.hr_size + 20
        if hr_image.width < min_size or hr_image.height < min_size:
            scale = max(min_size / hr_image.width, min_size / hr_image.height)
            new_size = (int(hr_image.width * scale), int(hr_image.height * scale))
            hr_image = hr_image.resize(new_size, Image.LANCZOS)
        
        # Apply transforms
        hr_tensor = self.transform(hr_image)
        
        # Create LR from HR using proper downsampling
        lr_tensor = F.interpolate(
            hr_tensor.unsqueeze(0), 
            size=(self.lr_size, self.lr_size), 
            mode='bicubic', 
            align_corners=False,
            antialias=True
        ).squeeze(0)
        
        return lr_tensor, hr_tensor


class ImprovedESRGANTrainer:
    """Improved ESRGAN Trainer with better training strategies"""
    
    def __init__(self, generator, discriminator, device='cuda', config=None):
        self.device = device
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.config = config or self._default_config()
        
        # Initialize losses
        self.criterionGAN = nn.BCEWithLogitsLoss()
        self.criterionL1 = nn.L1Loss()
        self.criterionMSE = nn.MSELoss()
        
        # Perceptual loss
        if self.config['use_perceptual']:
            self.perceptual_loss = PerceptualLoss().to(device)
            logger.info("Perceptual loss enabled")
        
        # Optimizers with improved settings
        self.optimizer_G = optim.Adam(
            self.generator.parameters(), 
            lr=self.config['lr_g'], 
            betas=(0.9, 0.99),
            weight_decay=1e-4
        )
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(), 
            lr=self.config['lr_d'], 
            betas=(0.9, 0.99),
            weight_decay=1e-4
        )
        
        # Learning rate schedulers
        self.scheduler_G = optim.lr_scheduler.MultiStepLR(
            self.optimizer_G, 
            milestones=[self.config['epochs']//2, int(self.config['epochs']*0.75)], 
            gamma=0.5
        )
        self.scheduler_D = optim.lr_scheduler.MultiStepLR(
            self.optimizer_D, 
            milestones=[self.config['epochs']//2, int(self.config['epochs']*0.75)], 
            gamma=0.5
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_psnr = 0
        self.training_history = []
        
        # Warmup settings
        self.warmup_epochs = self.config.get('warmup_epochs', 5)
        self.warmup_completed = False
        
        logger.info(f"Trainer initialized with config: {self.config}")
    
    def _default_config(self):
        return {
            'lr_g': 2e-4,
            'lr_d': 2e-4,
            'warmup_epochs': 5,
            'epochs': 200,
            'lambda_l1': 100,
            'lambda_perceptual': 1.0,
            'lambda_gan': 1.0,
            'use_perceptual': True,
            'discriminator_steps': 1,
            'generator_steps': 1,
            'gradient_penalty': 0.0,
        }
    
    def warmup_training(self, lr_imgs, hr_imgs):
        """Warmup phase with only L1 loss"""
        self.optimizer_G.zero_grad()
        
        fake_hr = self.generator(lr_imgs)
        
        # Only L1 loss during warmup
        loss_L1 = self.criterionL1(fake_hr, hr_imgs)
        loss_G = loss_L1
        
        loss_G.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
        self.optimizer_G.step()
        
        return {
            'loss_G': loss_G.item(),
            'loss_D': 0.0,
            'loss_GAN': 0.0,
            'loss_perceptual': 0.0,
            'loss_L1': loss_L1.item()
        }
    
    def train_step(self, lr_imgs, hr_imgs):
        batch_size = lr_imgs.size(0)
        lr_imgs = lr_imgs.to(self.device)
        hr_imgs = hr_imgs.to(self.device)
        
        # Warmup training (first few epochs with only L1 loss)
        if self.current_epoch < self.warmup_epochs:
            return self.warmup_training(lr_imgs, hr_imgs)
        
        # Train Generator
        self.optimizer_G.zero_grad()
        
        fake_hr = self.generator(lr_imgs)
        
        # Ensure same size
        if fake_hr.shape != hr_imgs.shape:
            fake_hr = F.interpolate(fake_hr, size=hr_imgs.shape[2:], mode='bilinear', align_corners=False)
        
        # Generator losses
        loss_L1 = self.criterionL1(fake_hr, hr_imgs)
        
        # Perceptual loss
        loss_perceptual = torch.tensor(0.0, device=self.device)
        if self.config['use_perceptual']:
            try:
                # Resize for VGG
                hr_vgg = F.interpolate(hr_imgs, size=(224, 224), mode='bilinear', align_corners=False)
                fake_hr_vgg = F.interpolate(fake_hr, size=(224, 224), mode='bilinear', align_corners=False)
                
                real_features = self.perceptual_loss(hr_vgg)
                fake_features = self.perceptual_loss(fake_hr_vgg)
                
                for real_feat, fake_feat in zip(real_features, fake_features):
                    loss_perceptual += self.criterionL1(fake_feat, real_feat)
                
                loss_perceptual /= len(real_features)  # Average over feature levels
            except Exception as e:
                logger.warning(f"Perceptual loss computation failed: {e}")
                loss_perceptual = torch.tensor(0.0, device=self.device)
        
        # Adversarial loss
        pred_fake = self.discriminator(fake_hr)
        real_labels = torch.ones_like(pred_fake, device=self.device) * 0.9  # Label smoothing
        loss_GAN = self.criterionGAN(pred_fake, real_labels)
        
        # Combined generator loss
        loss_G = (self.config['lambda_gan'] * loss_GAN + 
                 self.config['lambda_l1'] * loss_L1 + 
                 self.config['lambda_perceptual'] * loss_perceptual)
        
        loss_G.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)  # Gradient clipping
        self.optimizer_G.step()
        
        # Train Discriminator
        self.optimizer_D.zero_grad()
        
        # Real loss
        pred_real = self.discriminator(hr_imgs)
        real_labels = torch.ones_like(pred_real, device=self.device) * 0.9
        loss_D_real = self.criterionGAN(pred_real, real_labels)
        
        # Fake loss
        pred_fake = self.discriminator(fake_hr.detach())
        fake_labels = torch.zeros_like(pred_fake, device=self.device) + 0.1
        loss_D_fake = self.criterionGAN(pred_fake, fake_labels)
        
        # Combined discriminator loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        
        loss_D.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
        self.optimizer_D.step()
        
        self.global_step += 1
        
        return {
            'loss_G': loss_G.item(),
            'loss_D': loss_D.item(),
            'loss_GAN': loss_GAN.item(),
            'loss_perceptual': loss_perceptual.item(),
            'loss_L1': loss_L1.item()
        }
    
    def calculate_psnr(self, img1, img2):
        """Calculate PSNR between two images"""
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * torch.log10(1.0 / torch.sqrt(mse))
    
    def validate(self, val_loader):
        """Validation step"""
        self.generator.eval()
        total_psnr = 0
        total_loss = 0
        count = 0
        
        with torch.no_grad():
            for lr_imgs, hr_imgs in val_loader:
                lr_imgs = lr_imgs.to(self.device)
                hr_imgs = hr_imgs.to(self.device)
                
                fake_hr = self.generator(lr_imgs)
                
                if fake_hr.shape != hr_imgs.shape:
                    fake_hr = F.interpolate(fake_hr, size=hr_imgs.shape[2:], mode='bilinear', align_corners=False)
                
                # Calculate metrics
                psnr = self.calculate_psnr(fake_hr, hr_imgs)
                loss = self.criterionL1(fake_hr, hr_imgs)
                
                total_psnr += psnr.item()
                total_loss += loss.item()
                count += 1
        
        self.generator.train()
        
        avg_psnr = total_psnr / count
        avg_loss = total_loss / count
        
        return avg_psnr, avg_loss
    
    def save_checkpoint(self, epoch, path, is_best=False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'scheduler_G_state_dict': self.scheduler_G.state_dict(),
            'scheduler_D_state_dict': self.scheduler_D.state_dict(),
            'config': self.config,
            'training_history': self.training_history,
            'best_psnr': self.best_psnr
        }
        
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = path.replace('.pth', '_best.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved: {best_path}")
    
    def load_checkpoint(self, path):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        
        if 'scheduler_G_state_dict' in checkpoint:
            self.scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
            self.scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint.get('global_step', 0)
        self.training_history = checkpoint.get('training_history', [])
        self.best_psnr = checkpoint.get('best_psnr', 0)
        
        logger.info(f"Checkpoint loaded from epoch {self.current_epoch}")
        return self.current_epoch


def create_data_loaders(hr_dir, val_split=0.1, batch_size=4, hr_size=192):
    """Create training and validation data loaders"""
    
    # Get all HR files
    hr_path = Path(hr_dir)
    all_files = sorted(list(hr_path.glob("*.png")) + list(hr_path.glob("*.jpg")))
    
    # Split into train/val
    val_size = int(len(all_files) * val_split)
    random.shuffle(all_files)
    
    val_files = all_files[:val_size]
    train_files = all_files[val_size:]
    
    logger.info(f"Training files: {len(train_files)}, Validation files: {len(val_files)}")
    
    # Create temporary directories for validation (this is simplified - in practice you might want a different approach)
    val_dir = hr_path.parent / "temp_val"
    val_dir.mkdir(exist_ok=True)
    
    # Create datasets
    train_dataset = ImprovedDIV2KDataset(hr_dir, hr_size=hr_size, mode='train')
    val_dataset = ImprovedDIV2KDataset(hr_dir, hr_size=hr_size, mode='val')
    
    # Limit val dataset size for faster validation
    val_dataset.hr_files = val_dataset.hr_files[:min(50, len(val_dataset.hr_files))]
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader


def train_esrgan(config):
    """Main training function"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create models
    generator = RDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4)
    discriminator = Discriminator(in_nc=3, base_nf=64)
    
    # Count parameters
    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    logger.info(f"Generator parameters: {g_params:,}")
    logger.info(f"Discriminator parameters: {d_params:,}")
    
    # Create trainer
    trainer = ImprovedESRGANTrainer(generator, discriminator, device, config)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        config['hr_dir'], 
        val_split=0.1,
        batch_size=config['batch_size'],
        hr_size=config.get('hr_size', 192)
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if config.get('resume_from'):
        start_epoch = trainer.load_checkpoint(config['resume_from']) + 1
    
    # Create output directories
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(f"{config['output_dir']}/checkpoints", exist_ok=True)
    os.makedirs(f"{config['output_dir']}/samples", exist_ok=True)
    
    # Training loop
    logger.info(f"Starting training from epoch {start_epoch} to {config['epochs']}")
    
    for epoch in range(start_epoch, config['epochs']):
        trainer.current_epoch = epoch
        
        # Training phase
        epoch_losses = {'loss_G': [], 'loss_D': [], 'loss_GAN': [], 'loss_perceptual': [], 'loss_L1': []}
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        
        for batch_idx, (lr_imgs, hr_imgs) in enumerate(progress_bar):
            losses = trainer.train_step(lr_imgs, hr_imgs)
            
            for key in epoch_losses:
                epoch_losses[key].append(losses[key])
            
            # Update progress bar
            if batch_idx % 10 == 0:
                progress_bar.set_postfix({
                    'G_loss': f"{losses['loss_G']:.4f}",
                    'D_loss': f"{losses['loss_D']:.4f}",
                    'PSNR_est': f"{-10*math.log10(losses['loss_L1']) if losses['loss_L1'] > 0 else 0:.2f}"
                })
        
        # Calculate epoch averages
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
        
        # Validation
        if epoch % config.get('val_interval', 5) == 0:
            val_psnr, val_loss = trainer.validate(val_loader)
            logger.info(f"Validation - PSNR: {val_psnr:.2f} dB, Loss: {val_loss:.4f}")
            
            # Save best model
            if val_psnr > trainer.best_psnr:
                trainer.best_psnr = val_psnr
                is_best = True
            else:
                is_best = False
        else:
            val_psnr, val_loss = 0, 0
            is_best = False
        
        # Update learning rates
        trainer.scheduler_G.step()
        trainer.scheduler_D.step()
        
        # Log training info
        log_info = {
            'epoch': epoch + 1,
            'lr_g': trainer.optimizer_G.param_groups[0]['lr'],
            'lr_d': trainer.optimizer_D.param_groups[0]['lr'],
            **avg_losses,
            'val_psnr': val_psnr,
            'val_loss': val_loss
        }
        
        trainer.training_history.append(log_info)
        
        logger.info(f"Epoch {epoch+1} - G: {avg_losses['loss_G']:.4f}, D: {avg_losses['loss_D']:.4f}, "
                   f"L1: {avg_losses['loss_L1']:.4f}, Val PSNR: {val_psnr:.2f} dB")
        
        # Save checkpoint
        if (epoch + 1) % config.get('save_interval', 10) == 0 or is_best:
            checkpoint_path = f"{config['output_dir']}/checkpoints/esrgan_epoch_{epoch+1}.pth"
            trainer.save_checkpoint(epoch + 1, checkpoint_path, is_best)
        
        # Save sample images
        if (epoch + 1) % config.get('sample_interval', 20) == 0:
            save_sample_images(trainer.generator, val_loader, epoch + 1, config['output_dir'], device)
    
    logger.info("Training completed!")
    
    # Save final model
    final_path = f"{config['output_dir']}/checkpoints/esrgan_final.pth"
    trainer.save_checkpoint(config['epochs'], final_path)
    
    # Save training history
    with open(f"{config['output_dir']}/training_history.json", 'w') as f:
        json.dump(trainer.training_history, f, indent=2)
    
    return trainer


def save_sample_images(generator, val_loader, epoch, output_dir, device):
    """Save sample super-resolved images"""
    generator.eval()
    
    sample_dir = f"{output_dir}/samples/epoch_{epoch}"
    os.makedirs(sample_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, (lr_imgs, hr_imgs) in enumerate(val_loader):
            if i >= 5:  # Save only first 5 samples
                break
                
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            fake_hr = generator(lr_imgs)
            
            # Convert to images
            lr_img = transforms.ToPILImage()(lr_imgs[0].cpu())
            hr_img = transforms.ToPILImage()(hr_imgs[0].cpu())
            sr_img = transforms.ToPILImage()(torch.clamp(fake_hr[0], 0, 1).cpu())
            
            # Create comparison
            comparison = Image.new('RGB', (lr_img.width * 4 + hr_img.width + sr_img.width, max(hr_img.height, sr_img.height)))
            
            # Resize LR for comparison
            lr_resized = lr_img.resize((lr_img.width * 4, lr_img.height * 4), Image.NEAREST)
            
            comparison.paste(lr_resized, (0, 0))
            comparison.paste(sr_img, (lr_resized.width, 0))
            comparison.paste(hr_img, (lr_resized.width + sr_img.width, 0))
            
            comparison.save(f"{sample_dir}/sample_{i+1}.png")
    
    generator.train()


def main():
    parser = argparse.ArgumentParser(description='Train ESRGAN Super Resolution Model')
    parser.add_argument('--hr_dir', type=str, required=True, help='HR images directory')
    parser.add_argument('--output_dir', type=str, default='training_output', help='Output directory')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr_g', type=float, default=2e-4, help='Generator learning rate')
    parser.add_argument('--lr_d', type=float, default=2e-4, help='Discriminator learning rate')
    parser.add_argument('--hr_size', type=int, default=192, help='HR patch size')
    parser.add_argument('--lambda_l1', type=float, default=100, help='L1 loss weight')
    parser.add_argument('--lambda_perceptual', type=float, default=1.0, help='Perceptual loss weight')
    parser.add_argument('--use_perceptual', action='store_true', help='Use perceptual loss')
    parser.add_argument('--resume_from', type=str, help='Resume from checkpoint')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup epochs')
    
    args = parser.parse_args()
    
    config = {
        'hr_dir': args.hr_dir,
        'output_dir': args.output_dir,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr_g': args.lr_g,
        'lr_d': args.lr_d,
        'hr_size': args.hr_size,
        'lambda_l1': args.lambda_l1,
        'lambda_perceptual': args.lambda_perceptual,
        'lambda_gan': 1.0,
        'use_perceptual': args.use_perceptual,
        'resume_from': args.resume_from,
        'warmup_epochs': args.warmup_epochs,
        'save_interval': 10,
        'val_interval': 5,
        'sample_interval': 20,
        'discriminator_steps': 1,
        'generator_steps': 1,
    }
    
    logger.info(f"Training configuration: {config}")
    
    # Start training
    trainer = train_esrgan(config)
    
    logger.info("Training finished successfully!")


if __name__ == "__main__":
    # For direct execution without arguments
    if len(sys.argv) == 1:
        # Default configuration
        config = {
            'hr_dir': "archive/DIV2K_train_HR/DIV2K_train_HR",
            'output_dir': 'improved_training',
            'epochs': 200,
            'batch_size': 2,  # Reduced for memory
            'lr_g': 1e-4,     # Reduced learning rate
            'lr_d': 1e-4,
            'hr_size': 128,   # Smaller patches for stability
            'lambda_l1': 100,
            'lambda_perceptual': 1.0,
            'lambda_gan': 0.1,  # Reduced GAN loss initially
            'use_perceptual': True,
            'resume_from': None,
            'warmup_epochs': 10,  # Longer warmup
            'save_interval': 10,
            'val_interval': 5,
            'sample_interval': 20,
            'discriminator_steps': 1,
            'generator_steps': 1,
        }
        
        logger.info("Running with default configuration")
        trainer = train_esrgan(config)
    else:
        main()
