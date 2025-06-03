"""
Memory-efficient ESRGAN inference with image tiling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import sys
import os
import math

# Add current directory to path
sys.path.append('.')
from train import RDBNet

def tile_process(model, device, lr_tensor, tile_size=256, overlap=32):
    """Process large image by tiling to avoid memory issues"""
    
    b, c, h, w = lr_tensor.shape
    
    # If image is small enough, process directly
    if h <= tile_size and w <= tile_size:
        with torch.no_grad():
            return model(lr_tensor)
    
    # Calculate tiles
    stride = tile_size - overlap
    h_tiles = math.ceil((h - overlap) / stride)
    w_tiles = math.ceil((w - overlap) / stride)
    
    # Output tensor (4x larger)
    output = torch.zeros(b, c, h * 4, w * 4, device=device)
    
    print(f"Processing {h_tiles}x{w_tiles} tiles...")
    
    for i in range(h_tiles):
        for j in range(w_tiles):
            # Calculate tile boundaries
            h_start = i * stride
            h_end = min(h_start + tile_size, h)
            w_start = j * stride
            w_end = min(w_start + tile_size, w)
            
            # Extract tile
            tile = lr_tensor[:, :, h_start:h_end, w_start:w_end]
            
            # Process tile
            with torch.no_grad():
                tile_output = model(tile)
            
            # Calculate output position (4x coordinates)
            out_h_start = h_start * 4
            out_h_end = h_end * 4
            out_w_start = w_start * 4
            out_w_end = w_end * 4
            
            # Handle overlap blending
            if i == 0 and j == 0:
                # First tile - no blending
                output[:, :, out_h_start:out_h_end, out_w_start:out_w_end] = tile_output
            else:
                # Blend with existing output
                current = output[:, :, out_h_start:out_h_end, out_w_start:out_w_end]
                if current.sum() > 0:
                    # Simple averaging for overlap regions
                    output[:, :, out_h_start:out_h_end, out_w_start:out_w_end] = (current + tile_output) / 2
                else:
                    output[:, :, out_h_start:out_h_end, out_w_start:out_w_end] = tile_output
            
            print(f"  Tile {i+1}/{h_tiles}, {j+1}/{w_tiles} done")
    
    return output

def memory_efficient_enhance(input_path, output_path, model_path, max_size=512):
    """Memory-efficient image enhancement"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    try:
        model = RDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if 'generator_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['generator_state_dict'])
            print(f"✅ Loaded generator from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)
            print("✅ Loaded model weights")
            
        model = model.to(device)
        model.eval()
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Process image
    try:
        # Load input
        lr_image = Image.open(input_path).convert('RGB')
        original_size = lr_image.size
        print(f"📥 Input size: {original_size}")
        
        # Check if we need to resize for memory
        max_dim = max(original_size)
        if max_dim > max_size:
            scale = max_size / max_dim
            new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
            lr_image = lr_image.resize(new_size, Image.LANCZOS)
            print(f"🔄 Resized to: {lr_image.size} (to fit in memory)")
        
        # Convert to tensor
        transform = transforms.ToTensor()
        lr_tensor = transform(lr_image).unsqueeze(0).to(device)
        
        print(f"💾 Tensor size: {lr_tensor.shape}")
        
        # Process with tiling
        sr_tensor = tile_process(model, device, lr_tensor, tile_size=256, overlap=32)
        sr_tensor = torch.clamp(sr_tensor, 0, 1)
        
        # Convert back to image
        to_pil = transforms.ToPILImage()
        sr_image = to_pil(sr_tensor.squeeze(0).cpu())
        
        # If we resized, scale back up
        if max_dim > max_size:
            final_size = (original_size[0] * 4, original_size[1] * 4)
            sr_image = sr_image.resize(final_size, Image.LANCZOS)
            print(f"🔄 Final resize to: {final_size}")
        
        # Save result
        sr_image.save(output_path, quality=95)
        print(f"📤 Output size: {sr_image.size}")
        print(f"✅ Enhanced image saved: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return False

if __name__ == "__main__":
    # Usage
    input_img = "results_best_model/generated_sr/0801.png"
    output_img = "enhanced_image.jpg" 
    model_file = "improved_training/checkpoints/esrgan_epoch_86_best.pth"
    
    if len(sys.argv) >= 2:
        input_img = sys.argv[1]
    if len(sys.argv) >= 3:
        output_img = sys.argv[2]
    if len(sys.argv) >= 4:
        model_file = sys.argv[3]
    
    print(f"🚀 ESRGAN Memory-Efficient Enhance")
    print(f"Input: {input_img}")
    print(f"Output: {output_img}")
    print(f"Model: {model_file}")
    print("-" * 50)
    
    success = memory_efficient_enhance(input_img, output_img, model_file, max_size=512)
    
    if success:
        print("🎉 Enhancement completed successfully!")
    else:
        print("💥 Enhancement failed!")
