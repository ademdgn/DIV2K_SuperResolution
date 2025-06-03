"""
ESRGAN Super Resolution Model - Custom Implementation
4x upscaling for image enhancement
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from pathlib import Path


class DIV2KDataset(Dataset):
    def __init__(self, hr_dir, hr_size=128, scale_factor=4):
        self.hr_dir = Path(hr_dir)
        self.hr_size = hr_size
        self.lr_size = hr_size // scale_factor
        self.scale_factor = scale_factor
        
        self.hr_files = sorted(list(self.hr_dir.glob("*.png")))
        print(f"Found {len(self.hr_files)} images")
        
        self.transform = transforms.Compose([
            transforms.RandomCrop(self.hr_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        return len(self.hr_files)
    
    def __getitem__(self, idx):
        hr_path = self.hr_files[idx]
        hr_image = Image.open(hr_path).convert('RGB')
        
        seed = torch.randint(0, 2147483647, size=(1,)).item()
        torch.manual_seed(seed)
        hr_tensor = self.transform(hr_image)
        
        # Generate LR from HR
        lr_tensor = F.interpolate(
            hr_tensor.unsqueeze(0), 
            size=(self.lr_size, self.lr_size), 
            mode='bicubic', 
            align_corners=False
        ).squeeze(0)
        
        return lr_tensor, hr_tensor


class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        
    def forward(self, x):
        x1 = F.leaky_relu(self.conv1(x), 0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(torch.cat((x, x1), 1)), 0.2, True)
        x3 = F.leaky_relu(self.conv3(torch.cat((x, x1, x2), 1)), 0.2, True)
        x4 = F.leaky_relu(self.conv4(torch.cat((x, x1, x2, x3), 1)), 0.2, True)
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock(nf, gc)
        self.RDB2 = ResidualDenseBlock(nf, gc)
        self.RDB3 = ResidualDenseBlock(nf, gc)
        
    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RDBNet(nn.Module):
    """ESRGAN Generator with Residual Dense Blocks"""
    
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4):
        super(RDBNet, self).__init__()
        
        self.scale = scale
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        
        self.RRDB_trunk = nn.ModuleList()
        for _ in range(nb):
            self.RRDB_trunk.append(RRDB(nf, gc))
            
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1)
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)
        
    def forward(self, x):
        fea = self.conv_first(x)
        trunk = fea
        
        for rrdb in self.RRDB_trunk:
            trunk = rrdb(trunk)
            
        trunk = self.trunk_conv(trunk)
        fea = fea + trunk
        
        # 4x upscaling (2x + 2x)
        fea = F.leaky_relu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')), 0.2, True)
        fea = F.leaky_relu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')), 0.2, True)
        
        out = self.conv_last(F.leaky_relu(self.HRconv(fea), 0.2, True))
        
        return out


class Discriminator(nn.Module):
    """ESRGAN Discriminator"""
    
    def __init__(self, in_nc=3, base_nf=64):
        super(Discriminator, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_nc, base_nf, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(base_nf, base_nf, 4, 2, 1),
            nn.BatchNorm2d(base_nf),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(base_nf, base_nf * 2, 3, 1, 1),
            nn.BatchNorm2d(base_nf * 2),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(base_nf * 2, base_nf * 2, 4, 2, 1),
            nn.BatchNorm2d(base_nf * 2),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(base_nf * 2, base_nf * 4, 3, 1, 1),
            nn.BatchNorm2d(base_nf * 4),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(base_nf * 4, base_nf * 4, 4, 2, 1),
            nn.BatchNorm2d(base_nf * 4),
            nn.LeakyReLU(0.2, True),
            
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(base_nf * 4, 100),
            nn.LeakyReLU(0.2, True),
            nn.Linear(100, 1)
        )
        
    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        out = self.classifier(features)
        return out


class ESRGANTrainer:
    def __init__(self, generator, discriminator, device='cuda'):
        self.device = device
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        
        # Loss functions
        self.criterionGAN = nn.BCEWithLogitsLoss()
        self.criterionL1 = nn.L1Loss()
        
        # Optimizers
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=1e-4, betas=(0.9, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=1e-4, betas=(0.9, 0.999))
        
        # Schedulers
        self.scheduler_G = optim.lr_scheduler.StepLR(self.optimizer_G, step_size=50, gamma=0.5)
        self.scheduler_D = optim.lr_scheduler.StepLR(self.optimizer_D, step_size=50, gamma=0.5)
        
    def train_step(self, lr_imgs, hr_imgs):
        batch_size = lr_imgs.size(0)
        lr_imgs = lr_imgs.to(self.device)
        hr_imgs = hr_imgs.to(self.device)
        
        # Train Generator
        self.optimizer_G.zero_grad()
        
        fake_hr = self.generator(lr_imgs)
        
        # Ensure same size
        if fake_hr.shape != hr_imgs.shape:
            fake_hr = F.interpolate(fake_hr, size=hr_imgs.shape[2:], mode='bilinear', align_corners=False)
        
        # Generator losses
        pred_fake = self.discriminator(fake_hr)
        real_labels = torch.ones_like(pred_fake) * 0.9  # Label smoothing
        loss_GAN = self.criterionGAN(pred_fake, real_labels)
        
        loss_L1 = self.criterionL1(fake_hr, hr_imgs)
        
        # Combined loss
        loss_G = loss_GAN + 100 * loss_L1
        
        loss_G.backward()
        self.optimizer_G.step()
        
        # Train Discriminator
        self.optimizer_D.zero_grad()
        
        # Real loss
        pred_real = self.discriminator(hr_imgs)
        real_labels = torch.ones_like(pred_real) * 0.9
        loss_D_real = self.criterionGAN(pred_real, real_labels)
        
        # Fake loss
        pred_fake = self.discriminator(fake_hr.detach())
        fake_labels = torch.zeros_like(pred_fake) + 0.1
        loss_D_fake = self.criterionGAN(pred_fake, fake_labels)
        
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        
        loss_D.backward()
        self.optimizer_D.step()
        
        return {
            'loss_G': loss_G.item(),
            'loss_D': loss_D.item(),
            'loss_GAN': loss_GAN.item(),
            'loss_L1': loss_L1.item()
        }
    
    def save_checkpoint(self, epoch, path):
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict()
        }, path)


def train_esrgan(hr_dir, num_epochs=100, batch_size=4, save_interval=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset and DataLoader
    dataset = DIV2KDataset(hr_dir, hr_size=128, scale_factor=4)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Models
    generator = RDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4)
    discriminator = Discriminator(in_nc=3, base_nf=64)
    
    # Trainer
    trainer = ESRGANTrainer(generator, discriminator, device)
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_losses = {'loss_G': 0, 'loss_D': 0, 'loss_GAN': 0, 'loss_L1': 0}
        
        for i, (lr_imgs, hr_imgs) in enumerate(dataloader):
            losses = trainer.train_step(lr_imgs, hr_imgs)
            
            for key in epoch_losses:
                epoch_losses[key] += losses[key]
                
            if i % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{i}/{len(dataloader)}] "
                     f"G_loss: {losses['loss_G']:.4f} D_loss: {losses['loss_D']:.4f}")
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= len(dataloader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] Avg - G: {epoch_losses['loss_G']:.4f}, "
              f"D: {epoch_losses['loss_D']:.4f}, L1: {epoch_losses['loss_L1']:.4f}")
        
        # Update learning rates
        trainer.scheduler_G.step()
        trainer.scheduler_D.step()
        
        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            os.makedirs('checkpoints', exist_ok=True)
            trainer.save_checkpoint(epoch + 1, f'checkpoints/esrgan_epoch_{epoch+1}.pth')
            print(f"Model saved at epoch {epoch+1}")
    
    return trainer


def super_resolve_image(model_path, lr_image_path, output_path, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = RDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if 'generator_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['generator_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Process image
    lr_image = Image.open(lr_image_path).convert('RGB')
    lr_tensor = transforms.ToTensor()(lr_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        sr_tensor = model(lr_tensor)
        sr_tensor = torch.clamp(sr_tensor, 0, 1)
    
    sr_image = transforms.ToPILImage()(sr_tensor.squeeze(0).cpu())
    sr_image.save(output_path)
    
    return sr_image


if __name__ == "__main__":
    # Training
    HR_DIR = "archive/DIV2K_train_HR/DIV2K_train_HR"
    
    if os.path.exists(HR_DIR):
        trainer = train_esrgan(
            hr_dir=HR_DIR,
            num_epochs=50,
            batch_size=2,
            save_interval=10
        )
        print("Training completed!")
    else:
        print(f"Dataset directory not found: {HR_DIR}")
        print("Please download DIV2K dataset and extract to archive/")
