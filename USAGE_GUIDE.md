# 🚀 ESRGAN Usage Guide

## Quick Start

### 1. Setup Environment
```bash
git clone https://github.com/yourusername/ESRGAN-SuperRes
cd ESRGAN-SuperRes
pip install -r requirements.txt
```

### 2. Enhance Your Images
```bash
python inference.py --input your_image.jpg --output enhanced_image.jpg
```

## 📝 Detailed Usage

### Training Your Own Model

1. **Download Dataset**
   - Get DIV2K dataset from [here](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
   - Extract to `archive/DIV2K_train_HR/DIV2K_train_HR/`

2. **Start Training**
   ```bash
   python train.py
   ```

3. **Monitor Progress**
   - Checkpoints saved every 10 epochs in `checkpoints/`
   - Training logs printed to console

### Inference Options

```bash
python inference.py [OPTIONS]

Options:
  -i, --input PATH     Input image path (required)
  -o, --output PATH    Output image path (required)  
  -m, --model PATH     Model checkpoint path [default: checkpoints/esrgan_final.pth]
  --device TEXT        Device: auto/cuda/cpu [default: auto]
```

### Examples

**Basic Enhancement:**
```bash
python inference.py -i photo.jpg -o photo_4x.jpg
```

**Using Specific Model:**
```bash
python inference.py -i input.jpg -o output.jpg -m checkpoints/esrgan_epoch_50.pth
```

**Force CPU Usage:**
```bash
python inference.py -i input.jpg -o output.jpg --device cpu
```

## 🎛️ Configuration

### Training Parameters
Edit `train.py` to customize:

```python
# Dataset
hr_size = 128           # HR patch size
scale_factor = 4        # Upscaling factor

# Training
num_epochs = 100        # Training epochs
batch_size = 4          # Batch size
learning_rate = 1e-4    # Learning rate

# Model
nf = 64                 # Number of features
nb = 23                 # Number of RRDB blocks
gc = 32                 # Growth channels
```

## 🔧 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size in train.py
batch_size = 2  # or 1
```

**2. Model File Not Found**
```bash
# Check available models
ls checkpoints/
```

**3. Import Errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Performance Tips

- **GPU Memory**: Use smaller batch sizes if CUDA memory is limited
- **CPU Training**: Expect much slower training on CPU
- **Image Size**: Larger input images need more memory

## 📊 Model Performance

| Setting | PSNR | SSIM | Training Time |
|---------|------|------|---------------|
| Default | 25.3 | 0.777| ~12 hours     |
| Small   | 24.1 | 0.745| ~6 hours      |
| Large   | 26.8 | 0.801| ~24 hours     |

## 🎯 Best Practices

1. **Training**
   - Start with small batch size
   - Monitor GPU memory usage
   - Save checkpoints frequently

2. **Inference**
   - Use GPU for faster processing
   - Process images in batches for efficiency
   - Keep input images reasonable size

3. **Quality**
   - Train longer for better results
   - Use high-quality training data
   - Experiment with different loss weights

## 🤔 FAQ

**Q: How long does training take?**
A: ~12 hours on RTX 3080 for 100 epochs

**Q: What input formats are supported?**
A: JPG, PNG, and most common image formats

**Q: Can I change the upscaling factor?**
A: Currently supports 4x, but can be modified in the model architecture

**Q: Does it work on CPU?**
A: Yes, but much slower than GPU

**Q: How to improve results?**
A: Train longer, use more data, tune hyperparameters