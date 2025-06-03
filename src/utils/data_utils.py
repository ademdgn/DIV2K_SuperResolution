"""
Data utilities for ESRGAN Super Resolution
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
import torch.nn.functional as F
from typing import Tuple, List, Optional


class DIV2KDataset(Dataset):
    """DIV2K Dataset for Super Resolution"""
    
    def __init__(self, hr_dir, lr_dir=None, hr_size=128, scale_factor=4, mode='train'):
        """
        DIV2K Dataset sınıfı
        Args:
            hr_dir: Yüksek çözünürlük görüntülerin dizini
            lr_dir: Düşük çözünürlük görüntülerin dizini (None ise HR'den üretilir)
            hr_size: HR patch boyutu
            scale_factor: Ölçeklendirme faktörü (2x, 3x, 4x)
            mode: 'train' veya 'test'
        """
        self.hr_dir = Path(hr_dir)
        self.lr_dir = Path(lr_dir) if lr_dir else None
        self.hr_size = hr_size
        self.lr_size = hr_size // scale_factor
        self.scale_factor = scale_factor
        self.mode = mode
        
        # Dosya listelerini al
        self.hr_files = sorted(list(self.hr_dir.glob("*.png")) + list(self.hr_dir.glob("*.jpg")))
        if self.lr_dir:
            self.lr_files = sorted(list(self.lr_dir.glob("*.png")) + list(self.lr_dir.glob("*.jpg")))
            print(f"Found {len(self.hr_files)} HR images and {len(self.lr_files)} LR images")
        else:
            self.lr_files = []
            print(f"Found {len(self.hr_files)} High Resolution Files.")
            print("No LR directory provided - will generate LR images on-the-fly.")
        
        # Transform tanımları
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomCrop(self.hr_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        
    def __len__(self):
        return len(self.hr_files)
    
    def __getitem__(self, idx):
        # HR görüntüyü yükle
        hr_path = self.hr_files[idx]
        hr_image = Image.open(hr_path).convert('RGB')
        
        # LR görüntüyü yükle veya oluştur
        if self.lr_dir and idx < len(self.lr_files):
            lr_path = self.lr_files[idx]
            lr_image = Image.open(lr_path).convert('RGB')
        else:
            # LR directory verilmemiş veya dosya yok, HR'den oluştur
            lr_image = hr_image.resize(
                (hr_image.width // self.scale_factor, hr_image.height // self.scale_factor), 
                Image.BICUBIC
            )
        
        if self.mode == 'train':
            # Random crop için seed kullan
            seed = torch.randint(0, 2147483647, size=(1,)).item()
            
            # HR transform (crop ve flip)
            torch.manual_seed(seed)
            hr_tensor = self.transform(hr_image)
            
            # LR'den aynı bölgeyi al
            torch.manual_seed(seed)
            # Önce HR ile aynı crop'u yap
            hr_temp = self.transform(hr_image)
            
            # Sonra LR'e downscale et
            lr_tensor = F.interpolate(
                hr_temp.unsqueeze(0), 
                size=(self.lr_size, self.lr_size), 
                mode='bicubic', 
                align_corners=False
            ).squeeze(0)
        else:
            # Test modunda crop yok
            hr_tensor = self.transform(hr_image)
            lr_tensor = self.transform(lr_image)
        
        return lr_tensor, hr_tensor


def create_lr_images_from_hr(hr_dir: str, lr_dir: str, scale_factor: int = 4):
    """Create LR images from HR images by downscaling"""
    hr_path = Path(hr_dir)
    lr_path = Path(lr_dir)
    lr_path.mkdir(parents=True, exist_ok=True)
    
    hr_images = list(hr_path.glob("*.png")) + list(hr_path.glob("*.jpg"))
    
    print(f"Creating LR images from {len(hr_images)} HR images...")
    
    for hr_img_path in hr_images:
        # Load HR image
        hr_image = Image.open(hr_img_path).convert('RGB')
        
        # Create LR image
        lr_size = (hr_image.width // scale_factor, hr_image.height // scale_factor)
        lr_image = hr_image.resize(lr_size, Image.BICUBIC)
        
        # Save LR image
        lr_img_path = lr_path / hr_img_path.name
        lr_image.save(lr_img_path)
    
    print(f"LR images saved to: {lr_path}")


def calculate_model_size(model: torch.nn.Module) -> Tuple[int, float]:
    """Calculate model parameters and size in MB"""
    param_count = sum(p.numel() for p in model.parameters())
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    
    return param_count, size_mb


def split_dataset(dataset_dir: str, train_ratio: float = 0.8, val_ratio: float = 0.1):
    """Split dataset into train/val/test sets"""
    import shutil
    import random
    
    dataset_path = Path(dataset_dir)
    
    # Get all images
    images = list(dataset_path.glob("*.png")) + list(dataset_path.glob("*.jpg"))
    random.shuffle(images)
    
    # Calculate split indices
    total_images = len(images)
    train_end = int(total_images * train_ratio)
    val_end = train_end + int(total_images * val_ratio)
    
    # Create directories
    for split in ['train', 'val', 'test']:
        split_dir = dataset_path.parent / f"{dataset_path.name}_{split}"
        split_dir.mkdir(exist_ok=True)
    
    # Split images
    splits = {
        'train': images[:train_end],
        'val': images[train_end:val_end],
        'test': images[val_end:]
    }
    
    for split_name, split_images in splits.items():
        split_dir = dataset_path.parent / f"{dataset_path.name}_{split_name}"
        for img_path in split_images:
            shutil.copy2(img_path, split_dir / img_path.name)
        print(f"{split_name}: {len(split_images)} images")
    
    return splits


def visualize_results(lr_image: np.ndarray, sr_image: np.ndarray, hr_image: np.ndarray, 
                     output_path: str = None):
    """Visualize LR, SR, HR comparison"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # LR Image
    axes[0].imshow(lr_image)
    axes[0].set_title('Low Resolution (Input)')
    axes[0].axis('off')
    
    # SR Image
    axes[1].imshow(sr_image)
    axes[1].set_title('Super Resolution (Output)')
    axes[1].axis('off')
    
    # HR Image
    axes[2].imshow(hr_image)
    axes[2].set_title('High Resolution (Ground Truth)')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_comparison_grid(images: List[np.ndarray], titles: List[str], 
                          output_path: str = None, grid_size: Tuple[int, int] = None):
    """Create a grid of images for comparison"""
    import matplotlib.pyplot as plt
    
    if grid_size is None:
        n_images = len(images)
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
    else:
        rows, cols = grid_size
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    
    if rows == 1:
        axes = [axes]
    if cols == 1:
        axes = [[ax] for ax in axes]
    
    for i, (image, title) in enumerate(zip(images, titles)):
        row = i // cols
        col = i % cols
        
        if row < rows and col < cols:
            axes[row][col].imshow(image)
            axes[row][col].set_title(title)
            axes[row][col].axis('off')
    
    # Hide empty subplots
    for i in range(len(images), rows * cols):
        row = i // cols
        col = i % cols
        if row < rows and col < cols:
            axes[row][col].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array for visualization"""
    if tensor.dim() == 4:  # Batch dimension
        tensor = tensor.squeeze(0)
    
    # Convert from CHW to HWC
    if tensor.dim() == 3:
        tensor = tensor.permute(1, 2, 0)
    
    # Convert to numpy and clip values
    array = tensor.detach().cpu().numpy()
    array = np.clip(array, 0, 1)
    
    # Convert to 0-255 range if needed
    if array.max() <= 1.0:
        array = (array * 255).astype(np.uint8)
    
    return array


def numpy_to_tensor(array: np.ndarray, device: str = 'cpu') -> torch.Tensor:
    """Convert numpy array to tensor"""
    # Normalize to 0-1 range if needed
    if array.max() > 1.0:
        array = array.astype(np.float32) / 255.0
    
    # Convert from HWC to CHW
    if array.ndim == 3:
        array = array.transpose(2, 0, 1)
    
    # Convert to tensor
    tensor = torch.from_numpy(array).float()
    
    # Add batch dimension if needed
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    
    return tensor.to(device)


def save_checkpoint(model, optimizer, scheduler, epoch, loss, filepath):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    """Load training checkpoint"""
    checkpoint = torch.load(filepath)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint.get('loss', 0)
