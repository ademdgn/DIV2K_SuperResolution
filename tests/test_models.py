"""
Unit tests for ESRGAN model components
"""

import unittest
import torch
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.models.esrgan import RDBNet, Discriminator, PerceptualLoss, ResidualDenseBlock, RRDB
from src.evaluation.metrics import SuperResolutionMetrics
from src.utils.data_utils import DIV2KDataset


class TestESRGANComponents(unittest.TestCase):
    """Test ESRGAN model components"""
    
    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = 2
        self.lr_size = 32
        self.hr_size = 128
        self.scale_factor = 4
        
    def test_residual_dense_block(self):
        """Test Residual Dense Block"""
        rdb = ResidualDenseBlock(nf=64, gc=32)
        
        # Test input/output shapes
        x = torch.randn(self.batch_size, 64, self.hr_size, self.hr_size)
        output = rdb(x)
        
        self.assertEqual(output.shape, x.shape)
        self.assertFalse(torch.allclose(output, x))  # Should modify input
        
    def test_rrdb(self):
        """Test Residual in Residual Dense Block"""
        rrdb = RRDB(nf=64, gc=32)
        
        x = torch.randn(self.batch_size, 64, self.hr_size, self.hr_size)
        output = rrdb(x)
        
        self.assertEqual(output.shape, x.shape)
        
    def test_generator(self):
        """Test Generator (RDBNet)"""
        generator = RDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4)
        
        # Test with LR input
        lr_input = torch.randn(self.batch_size, 3, self.lr_size, self.lr_size)
        sr_output = generator(lr_input)
        
        # Check output dimensions
        expected_shape = (self.batch_size, 3, self.lr_size * 4, self.lr_size * 4)
        self.assertEqual(sr_output.shape, expected_shape)
        
        # Check output range (should be reasonable for image data)
        self.assertTrue(sr_output.min() >= -2.0)  # Allow some flexibility
        self.assertTrue(sr_output.max() <= 2.0)
        
    def test_discriminator(self):
        """Test Discriminator"""
        discriminator = Discriminator(in_nc=3, base_nf=64)
        
        # Test with HR input
        hr_input = torch.randn(self.batch_size, 3, self.hr_size, self.hr_size)
        output = discriminator(hr_input)
        
        # Should output single value per image
        expected_shape = (self.batch_size, 1)
        self.assertEqual(output.shape, expected_shape)
        
    def test_perceptual_loss(self):
        """Test Perceptual Loss"""
        perceptual_loss = PerceptualLoss()
        
        # Test with image tensors [0, 1]
        img = torch.rand(self.batch_size, 3, 224, 224)  # VGG expects 224x224
        features = perceptual_loss(img)
        
        # Should return list of feature maps
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 4)  # 4 feature levels
        
    def test_model_parameters(self):
        """Test model parameter counts"""
        generator = RDBNet()
        discriminator = Discriminator()
        
        # Count parameters
        gen_params = sum(p.numel() for p in generator.parameters())
        disc_params = sum(p.numel() for p in discriminator.parameters())
        
        # Generator should have reasonable number of parameters
        self.assertGreater(gen_params, 1e6)  # At least 1M parameters
        self.assertLess(gen_params, 50e6)    # Less than 50M parameters
        
        # Discriminator should be smaller
        self.assertGreater(disc_params, 1e5)  # At least 100K parameters
        self.assertLess(disc_params, 10e6)    # Less than 10M parameters


class TestMetrics(unittest.TestCase):
    """Test evaluation metrics"""
    
    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.metrics = SuperResolutionMetrics(self.device)
        
        # Create test images
        self.test_image_1 = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        self.test_image_2 = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        
    def test_psnr_calculation(self):
        """Test PSNR calculation"""
        # Test identical images
        psnr_identical = self.metrics.calculate_psnr(self.test_image_1, self.test_image_1)
        self.assertEqual(psnr_identical, float('inf'))
        
        # Test different images
        psnr_different = self.metrics.calculate_psnr(self.test_image_1, self.test_image_2)
        self.assertIsInstance(psnr_different, float)
        self.assertGreater(psnr_different, 0)
        
    def test_ssim_calculation(self):
        """Test SSIM calculation"""
        # Test identical images
        ssim_identical = self.metrics.calculate_ssim(self.test_image_1, self.test_image_1)
        self.assertAlmostEqual(ssim_identical, 1.0, places=5)
        
        # Test different images
        ssim_different = self.metrics.calculate_ssim(self.test_image_1, self.test_image_2)
        self.assertIsInstance(ssim_different, float)
        self.assertGreaterEqual(ssim_different, -1.0)
        self.assertLessEqual(ssim_different, 1.0)
        
    def test_mse_calculation(self):
        """Test MSE calculation"""
        # Test identical images
        mse_identical = self.metrics.calculate_mse(self.test_image_1, self.test_image_1)
        self.assertEqual(mse_identical, 0.0)
        
        # Test different images
        mse_different = self.metrics.calculate_mse(self.test_image_1, self.test_image_2)
        self.assertIsInstance(mse_different, float)
        self.assertGreaterEqual(mse_different, 0.0)
        
    def test_evaluate_single_image_pair(self):
        """Test single image pair evaluation"""
        metrics = self.metrics.evaluate_single_image_pair(self.test_image_1, self.test_image_2)
        
        # Check that all expected metrics are present
        expected_metrics = ['psnr', 'ssim', 'mse', 'mae', 'rmse']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], float)


class TestDataUtils(unittest.TestCase):
    """Test data utility functions"""
    
    def test_tensor_to_numpy_conversion(self):
        """Test tensor to numpy conversion"""
        from src.utils.data_utils import tensor_to_numpy, numpy_to_tensor
        
        # Test tensor to numpy
        tensor = torch.rand(3, 64, 64)  # CHW format
        array = tensor_to_numpy(tensor)
        
        self.assertEqual(array.shape, (64, 64, 3))  # Should be HWC
        self.assertEqual(array.dtype, np.uint8)
        
        # Test numpy to tensor
        tensor_back = numpy_to_tensor(array)
        self.assertEqual(tensor_back.shape, (1, 3, 64, 64))  # Should be BCHW
        
    def test_model_size_calculation(self):
        """Test model size calculation"""
        from src.utils.data_utils import calculate_model_size
        
        model = RDBNet()
        param_count, size_mb = calculate_model_size(model)
        
        self.assertIsInstance(param_count, int)
        self.assertIsInstance(size_mb, float)
        self.assertGreater(param_count, 0)
        self.assertGreater(size_mb, 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline"""
    
    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def test_end_to_end_inference(self):
        """Test complete inference pipeline"""
        # Create model
        model = RDBNet().to(self.device)
        model.eval()
        
        # Create dummy input
        lr_input = torch.rand(1, 3, 32, 32).to(self.device)
        
        # Run inference
        with torch.no_grad():
            sr_output = model(lr_input)
            sr_output = torch.clamp(sr_output, 0, 1)
        
        # Check output
        self.assertEqual(sr_output.shape, (1, 3, 128, 128))
        self.assertTrue(torch.all(sr_output >= 0))
        self.assertTrue(torch.all(sr_output <= 1))
        
    def test_training_step_shapes(self):
        """Test that training step produces correct shapes"""
        generator = RDBNet().to(self.device)
        discriminator = Discriminator().to(self.device)
        
        # Create dummy data
        lr_imgs = torch.rand(2, 3, 32, 32).to(self.device)
        hr_imgs = torch.rand(2, 3, 128, 128).to(self.device)
        
        # Forward pass
        fake_hr = generator(lr_imgs)
        pred_fake = discriminator(fake_hr)
        pred_real = discriminator(hr_imgs)
        
        # Check shapes
        self.assertEqual(fake_hr.shape, hr_imgs.shape)
        self.assertEqual(pred_fake.shape, (2, 1))
        self.assertEqual(pred_real.shape, (2, 1))


if __name__ == '__main__':
    # Run tests
    unittest.main()
