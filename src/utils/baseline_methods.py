"""
Baseline methods for Super Resolution comparison
"""

import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn.functional as F


class BicubicUpsampler:
    """Bicubic interpolation baseline"""
    
    def __init__(self, scale_factor: int = 4):
        self.scale_factor = scale_factor
    
    def upsample_image(self, image: np.ndarray) -> np.ndarray:
        """Upsample single image using bicubic interpolation"""
        h, w = image.shape[:2]
        new_h, new_w = h * self.scale_factor, w * self.scale_factor
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    def generate_sr_images(self, hr_dir: str, output_dir: str):
        """Generate SR images for entire directory"""
        hr_path = Path(hr_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        hr_images = list(hr_path.glob("*.png")) + list(hr_path.glob("*.jpg"))
        
        print(f"Generating bicubic SR images for {len(hr_images)} images...")
        
        for hr_img_path in tqdm(hr_images):
            # Load HR image
            hr_image = np.array(Image.open(hr_img_path).convert('RGB'))
            
            # Create LR image
            lr_image = cv2.resize(hr_image, 
                                (hr_image.shape[1] // self.scale_factor, 
                                 hr_image.shape[0] // self.scale_factor), 
                                interpolation=cv2.INTER_CUBIC)
            
            # Upsample to SR
            sr_image = self.upsample_image(lr_image)
            
            # Save SR image
            sr_pil = Image.fromarray(sr_image)
            sr_pil.save(output_path / hr_img_path.name)


class BilinearUpsampler:
    """Bilinear interpolation baseline"""
    
    def __init__(self, scale_factor: int = 4):
        self.scale_factor = scale_factor
    
    def upsample_image(self, image: np.ndarray) -> np.ndarray:
        """Upsample single image using bilinear interpolation"""
        h, w = image.shape[:2]
        new_h, new_w = h * self.scale_factor, w * self.scale_factor
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    def generate_sr_images(self, hr_dir: str, output_dir: str):
        """Generate SR images for entire directory"""
        hr_path = Path(hr_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        hr_images = list(hr_path.glob("*.png")) + list(hr_path.glob("*.jpg"))
        
        print(f"Generating bilinear SR images for {len(hr_images)} images...")
        
        for hr_img_path in tqdm(hr_images):
            # Load HR image
            hr_image = np.array(Image.open(hr_img_path).convert('RGB'))
            
            # Create LR image
            lr_image = cv2.resize(hr_image, 
                                (hr_image.shape[1] // self.scale_factor, 
                                 hr_image.shape[0] // self.scale_factor), 
                                interpolation=cv2.INTER_LINEAR)
            
            # Upsample to SR
            sr_image = self.upsample_image(lr_image)
            
            # Save SR image
            sr_pil = Image.fromarray(sr_image)
            sr_pil.save(output_path / hr_img_path.name)


class LanczosUpsampler:
    """Lanczos interpolation baseline"""
    
    def __init__(self, scale_factor: int = 4):
        self.scale_factor = scale_factor
    
    def upsample_image(self, image: np.ndarray) -> np.ndarray:
        """Upsample single image using Lanczos interpolation"""
        h, w = image.shape[:2]
        new_h, new_w = h * self.scale_factor, w * self.scale_factor
        
        # Convert to PIL for Lanczos
        pil_image = Image.fromarray(image)
        upsampled_pil = pil_image.resize((new_w, new_h), Image.LANCZOS)
        return np.array(upsampled_pil)
    
    def generate_sr_images(self, hr_dir: str, output_dir: str):
        """Generate SR images for entire directory"""
        hr_path = Path(hr_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        hr_images = list(hr_path.glob("*.png")) + list(hr_path.glob("*.jpg"))
        
        print(f"Generating Lanczos SR images for {len(hr_images)} images...")
        
        for hr_img_path in tqdm(hr_images):
            # Load HR image
            hr_image = np.array(Image.open(hr_img_path).convert('RGB'))
            
            # Create LR image
            lr_pil = Image.fromarray(hr_image)
            lr_pil = lr_pil.resize((hr_image.shape[1] // self.scale_factor, 
                                   hr_image.shape[0] // self.scale_factor), 
                                  Image.LANCZOS)
            lr_image = np.array(lr_pil)
            
            # Upsample to SR
            sr_image = self.upsample_image(lr_image)
            
            # Save SR image
            sr_pil = Image.fromarray(sr_image)
            sr_pil.save(output_path / hr_img_path.name)
