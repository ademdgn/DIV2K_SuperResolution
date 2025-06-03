"""
ESRGAN Image Super Resolution - Inference Script
Upscale your images by 4x using trained model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import argparse
import os


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
        
        fea = F.leaky_relu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')), 0.2, True)
        fea = F.leaky_relu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')), 0.2, True)
        
        out = self.conv_last(F.leaky_relu(self.HRconv(fea), 0.2, True))
        
        return out


def load_model(model_path, device='cuda'):
    """Load trained ESRGAN model"""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    model = RDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'generator_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['generator_state_dict'])
            print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)
            
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
        
    model = model.to(device)
    model.eval()
    
    return model, device


def super_resolve_image(model, device, input_path, output_path):
    """Super resolve single image"""
    
    try:
        # Load and preprocess image
        lr_image = Image.open(input_path).convert('RGB')
        original_size = lr_image.size
        
        transform = transforms.ToTensor()
        lr_tensor = transform(lr_image).unsqueeze(0).to(device)
        
        print(f"Input image size: {original_size}")
        print(f"Processing...")
        
        # Super resolution
        with torch.no_grad():
            sr_tensor = model(lr_tensor)
            sr_tensor = torch.clamp(sr_tensor, 0, 1)
        
        # Convert back to PIL image
        to_pil = transforms.ToPILImage()
        sr_image = to_pil(sr_tensor.squeeze(0).cpu())
        
        # Save result
        sr_image.save(output_path)
        
        print(f"Output image size: {sr_image.size}")
        print(f"Super-resolved image saved: {output_path}")
        
        return sr_image
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='ESRGAN Super Resolution')
    parser.add_argument('--input', '-i', type=str, required=True, help='Input image path')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output image path')
    parser.add_argument('--model', '-m', type=str, default='checkpoints/esrgan_final.pth', 
                       help='Model checkpoint path')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Check input file
    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        return
    
    # Check model file
    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        print("Available models:")
        if os.path.exists('checkpoints'):
            for f in os.listdir('checkpoints'):
                if f.endswith('.pth'):
                    print(f"  checkpoints/{f}")
        return
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load model
    model, device = load_model(args.model, device)
    if model is None:
        return
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Super resolve
    result = super_resolve_image(model, device, args.input, args.output)
    
    if result:
        print(" Super resolution completed successfully!")
    else:
        print(" Super resolution failed!")


if __name__ == "__main__":
    main()
