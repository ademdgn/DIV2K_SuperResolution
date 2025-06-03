# CUSTOM ESRGAN Super Resolution - 4x Image Enhancement 

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-v1.9+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A **custom implementation** of ESRGAN (Enhanced Super-Resolution Generative Adversarial Networks) that upscales images by **4x** while preserving details and enhancing quality.

## 🎯 What is this?

This project implements a **custom GAN model** that can take low-resolution images and enhance them to **4x higher resolution** with improved details and sharpness. Built from scratch using PyTorch.

## ✨ Features

- **4x Super Resolution**: Transform 256x256 images to 1024x1024
- **Custom ESRGAN Architecture**: Residual Dense Blocks with 23 RRDB layers
- **GAN Training**: Generator + Discriminator for realistic results
- **Easy to Use**: Simple inference script for single images
- **GPU Accelerated**: CUDA support for fast processing

## 🔥 Results

### Before vs After (4x Enhancement)

| Original (Low Resolution) | Enhanced (4x Super Resolution) |
|:-------------------------:|:------------------------------:|
| ![LR Image](demo/input_low_res.jpg) | ![SR Image](demo/output_super_res.jpg) |
| 256 × 256 pixels | **1024 × 1024 pixels** |

| Original | Enhanced |
|:--------:|:--------:|
| ![Input 1](demo/before_1.jpg) | ![Output 1](demo/after_1.jpg) |
| ![Input 2](demo/before_2.jpg) | ![Output 2](demo/after_2.jpg) |

*My custom ESRGAN model successfully enhances details, textures, and overall image quality!*

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/ESRGAN-SuperRes
cd ESRGAN-SuperRes
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Pre-trained Model
```bash
# Download my trained model (if available)
# Or train your own using the training script
```

### 4. Enhance Your Images!
```bash
python inference.py --input your_image.jpg --output enhanced_image.jpg
```

## 📁 Project Structure

```
ESRGAN-SuperRes/
│
├── train.py              # Training script
├── inference.py          # Image enhancement script
├── requirements.txt      # Dependencies
├── README.md             # This file
│
├── demo/                 # Demo images
│   ├── before_1.jpg
│   ├── after_1.jpg
│   └── ...
│
├── checkpoints/          # Saved models
│   └── esrgan_final.pth
│
└── src/                  # Source code (if organized)
    ├── models/
    └── utils/
```

## 🧠 Model Architecture

My custom ESRGAN implementation includes:

- **Generator**: RDBNet with 23 Residual-in-Residual Dense Blocks (RRDB)
- **Discriminator**: Multi-layer CNN with BatchNorm and LeakyReLU
- **Loss Functions**: Adversarial Loss + L1 Loss for stable training
- **Training Strategy**: Progressive training with learning rate scheduling

## 💻 Usage

### Enhance Single Image
```bash
python inference.py -i input.jpg -o output.jpg -m checkpoints/esrgan_final.pth
```

### Train Your Own Model
```bash
python train.py
```

**Note**: You'll need the DIV2K dataset for training. Download from [here](https://data.vision.ee.ethz.ch/cvl/DIV2K/).

## 🔧 Configuration

Edit the training parameters in `train.py`:

```python
# Training settings
num_epochs = 100
batch_size = 4
learning_rate = 1e-4
save_interval = 10
```

## 📊 Performance

| Metric | Score |
|--------|-------|
| PSNR   | 25.3 dB |
| SSIM   | 0.777 |
| Training Time | ~12 hours (RTX 3080) |
| Inference Speed | ~2 seconds/image |

## 🎮 Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA-capable GPU (recommended)
- 8GB+ GPU memory for training
- 2GB+ for inference

## 📝 Implementation Details

This is a **from-scratch implementation** featuring:

1. **Custom Dataset Loader**: Handles DIV2K high-resolution images
2. **RRDB Architecture**: Dense connections for better feature extraction
3. **GAN Training Loop**: Alternating Generator/Discriminator optimization
4. **Loss Balancing**: Carefully tuned loss weights for stability
5. **Progressive Upsampling**: 2x + 2x for 4x total enhancement

## 🤝 Contributing

Feel free to contribute! Areas for improvement:
- Add more loss functions (perceptual, SSIM)
- Implement different upsampling factors (2x, 8x)
- Add model quantization for mobile deployment
- Create web interface

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by the original [ESRGAN paper](https://arxiv.org/abs/1809.00219)
- DIV2K dataset for training data
- PyTorch community for excellent documentation

## 📞 Contact

If you have questions about this implementation:
- Open an issue on GitHub
- Check the code comments for technical details

---

**Made with ❤️ and lots of GPU hours**

⭐ **Star this repo if you found it helpful!** ⭐