"""
Quick inference script - simplified version
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import sys
import os

# Add current directory to path
sys.path.append('.')
from train import RDBNet

def quick_enhance(input_path, output_path, model_path):
    """Quick image enhancement"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    try:
        model = RDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4)
        
        # Load checkpoint with compatibility fix
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if 'generator_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['generator_state_dict'])
            print(f" Loaded generator from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)
            print(" Loaded model weights")
            
        model = model.to(device)
        model.eval()
        
    except Exception as e:
        print(f" Error loading model: {e}")
        return False
    
    # Process image
    try:
        # Load input
        lr_image = Image.open(input_path).convert('RGB')
        print(f" Input size: {lr_image.size}")
        
        # Convert to tensor
        transform = transforms.ToTensor()
        lr_tensor = transform(lr_image).unsqueeze(0).to(device)
        
        # Generate super-resolution
        with torch.no_grad():
            sr_tensor = model(lr_tensor)
            sr_tensor = torch.clamp(sr_tensor, 0, 1)
        
        # Convert back to image
        to_pil = transforms.ToPILImage()
        sr_image = to_pil(sr_tensor.squeeze(0).cpu())
        
        # Save result
        sr_image.save(output_path, quality=95)
        print(f" Output size: {sr_image.size}")
        print(f" Enhanced image saved: {output_path}")
        
        return True
        
    except Exception as e:
        print(f" Error processing image: {e}")
        return False

if __name__ == "__main__":
    # Simple usage
    input_img = "test_images/airplane00.jpg"
    output_img = "enhanced_image.jpg" 
    model_file = "improved_training/checkpoints/esrgan_epoch_86_best.pth"
    
    if len(sys.argv) >= 2:
        input_img = sys.argv[1]
    if len(sys.argv) >= 3:
        output_img = sys.argv[2]
    if len(sys.argv) >= 4:
        model_file = sys.argv[3]
    
    print(f" ESRGAN Quick Enhance")
    print(f"Input: {input_img}")
    print(f"Output: {output_img}")
    print(f"Model: {model_file}")
    print("-" * 50)
    
    success = quick_enhance(input_img, output_img, model_file)
    
    if success:
        print(" Enhancement completed successfully!")
    else:
        print(" Enhancement failed!")
