"""
ESRGAN Model Definitions
"""

from .esrgan import RDBNet, Discriminator, PerceptualLoss, ResidualDenseBlock, RRDB

__all__ = ['RDBNet', 'Discriminator', 'PerceptualLoss', 'ResidualDenseBlock', 'RRDB']
