"""
ESRGAN Model Definitions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super(ResidualDenseBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(nf, gc, kernel_size=3, stride=1, padding=1)
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
    """ESRGAN Generator Network"""
    
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4):
        super(RDBNet, self).__init__()
        
        self.scale = scale
        
        self.conv_first = nn.Conv2d(in_nc, nf, kernel_size=3, stride=1, padding=1)
        
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


class Discriminator(nn.Module):
    """ESRGAN Discriminator Network"""
    
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
