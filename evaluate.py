"""
Model Evaluation Script for ESRGAN Super Resolution
Run comprehensive evaluation on trained models
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.models.esrgan import RDBNet
    from src.evaluation.metrics import SuperResolutionMetrics, BenchmarkSuite, ModelComparator
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(current_dir, 'src'))
    
    from models.esrgan import RDBNet
    from evaluation.metrics import SuperResolutionMetrics, BenchmarkSuite, ModelComparator
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm


def load_model(model_path: str, device: str = 'cuda'):
    """Load trained ESRGAN model"""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = RDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4)
    model = model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if 'generator_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['generator_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model, device


def generate_sr_images(model, hr_dir: str, output_dir: str, device: str = 'cuda'):
    """Generate super-resolved images from HR test set"""
    hr_path = Path(hr_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get HR images
    hr_images = sorted(list(hr_path.glob("*.png")) + list(hr_path.glob("*.jpg")) + list(hr_path.glob("*.jpeg")))
    
    transform = transforms.ToTensor()
    to_pil = transforms.ToPILImage()
    
    print(f"Generating SR images for {len(hr_images)} test images...")
    
    inference_times = []
    
    for hr_img_path in tqdm(hr_images):
        # Load HR image
        hr_image = Image.open(hr_img_path).convert('RGB')
        
        # Create LR image (downscale by 4x)
        lr_image = hr_image.resize((hr_image.width // 4, hr_image.height // 4), Image.BICUBIC)
        
        # Convert to tensor
        lr_tensor = transform(lr_image).unsqueeze(0).to(device)
        
        # Measure inference time
        if device == 'cuda':
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            with torch.no_grad():
                sr_tensor = model(lr_tensor)
                sr_tensor = torch.clamp(sr_tensor, 0, 1)
            end_event.record()
            
            torch.cuda.synchronize()
            inference_time = start_event.elapsed_time(end_event)
        else:
            import time
            start_time = time.time()
            with torch.no_grad():
                sr_tensor = model(lr_tensor)
                sr_tensor = torch.clamp(sr_tensor, 0, 1)
            inference_time = (time.time() - start_time) * 1000
        
        inference_times.append(inference_time)
        
        # Convert back to PIL and save
        sr_image = to_pil(sr_tensor.squeeze(0).cpu())
        sr_image.save(output_path / hr_img_path.name)
    
    print(f"Average inference time: {np.mean(inference_times):.2f} ms")
    print(f"SR images saved to: {output_path}")
    
    return inference_times


def evaluate_model(model_path: str, test_hr_dir: str, output_dir: str = "evaluation_results"):
    """Run complete model evaluation"""
    
    # Setup
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model, device = load_model(model_path, 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate SR images
    sr_dir = output_path / "generated_sr"
    inference_times = generate_sr_images(model, test_hr_dir, str(sr_dir), device)
    
    # Initialize metrics calculator
    metrics_calc = SuperResolutionMetrics(device)
    
    # Evaluate generated images
    print("Calculating metrics...")
    results = metrics_calc.evaluate_dataset(str(sr_dir), test_hr_dir, str(output_path))
    
    # Add inference statistics
    inference_stats = {
        'mean_inference_time_ms': np.mean(inference_times),
        'std_inference_time_ms': np.std(inference_times),
        'min_inference_time_ms': np.min(inference_times),
        'max_inference_time_ms': np.max(inference_times),
        'total_images': len(inference_times)
    }
    
    results['inference_stats'] = inference_stats
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    stats = results['summary_stats']['mean']
    print(f"PSNR:  {stats['psnr']:.2f} ± {results['summary_stats']['std']['psnr']:.2f} dB")
    print(f"SSIM:  {stats['ssim']:.3f} ± {results['summary_stats']['std']['ssim']:.3f}")
    print(f"MSE:   {stats['mse']:.2f} ± {results['summary_stats']['std']['mse']:.2f}")
    print(f"MAE:   {stats['mae']:.2f} ± {results['summary_stats']['std']['mae']:.2f}")
    print(f"RMSE:  {stats['rmse']:.2f} ± {results['summary_stats']['std']['rmse']:.2f}")
    
    if stats.get('lpips', -1) != -1:
        print(f"LPIPS: {stats['lpips']:.3f} ± {results['summary_stats']['std']['lpips']:.3f}")
    
    print(f"\nInference Time: {inference_stats['mean_inference_time_ms']:.2f} ± {inference_stats['std_inference_time_ms']:.2f} ms")
    print(f"Total Images Processed: {inference_stats['total_images']}")
    
    print(f"\nDetailed results saved to: {output_path}")
    print("="*50)
    
    return results


def compare_with_baselines(model_path: str, test_hr_dir: str, output_dir: str = "baseline_comparison"):
    """Compare ESRGAN with baseline methods"""
    from src.utils.baseline_methods import BicubicUpsampler, BilinearUpsampler
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate results for all methods
    methods = {
        'ESRGAN': model_path,
        'Bicubic': BicubicUpsampler(),
        'Bilinear': BilinearUpsampler()
    }
    
    model_results = {}
    
    for method_name, method in methods.items():
        method_output_dir = output_path / f"{method_name}_results"
        method_output_dir.mkdir(exist_ok=True)
        
        if method_name == 'ESRGAN':
            # Use our trained model
            model, device = load_model(method, 'cuda' if torch.cuda.is_available() else 'cpu')
            generate_sr_images(model, test_hr_dir, str(method_output_dir), device)
        else:
            # Use baseline method
            method.generate_sr_images(test_hr_dir, str(method_output_dir))
        
        model_results[method_name] = str(method_output_dir)
    
    # Compare all methods
    comparator = ModelComparator('cuda' if torch.cuda.is_available() else 'cpu')
    comparison_df = comparator.compare_models(model_results, test_hr_dir, str(output_path))
    
    print("\n" + "="*60)
    print("BASELINE COMPARISON RESULTS")
    print("="*60)
    print(comparison_df.round(3))
    print("="*60)
    
    return comparison_df


def run_benchmark_suite(model_path: str, test_hr_dir: str, output_dir: str = "benchmark_results"):
    """Run complete benchmark suite"""
    benchmark = BenchmarkSuite('cuda' if torch.cuda.is_available() else 'cpu')
    results = benchmark.run_full_benchmark(model_path, test_hr_dir, output_dir)
    
    print(f"\nComplete benchmark results saved to: {output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate ESRGAN Super Resolution Model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model (.pth file)')
    parser.add_argument('--test_dir', type=str, required=True,
                       help='Directory containing HR test images')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--mode', type=str, choices=['evaluate', 'compare', 'benchmark'], 
                       default='evaluate',
                       help='Evaluation mode: evaluate, compare, or benchmark')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for inference')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    print(f"Model: {args.model_path}")
    print(f"Test directory: {args.test_dir}")
    print(f"Mode: {args.mode}")
    
    if args.mode == 'evaluate':
        results = evaluate_model(args.model_path, args.test_dir, args.output_dir)
        
    elif args.mode == 'compare':
        results = compare_with_baselines(args.model_path, args.test_dir, args.output_dir)
        
    elif args.mode == 'benchmark':
        results = run_benchmark_suite(args.model_path, args.test_dir, args.output_dir)
    
    print(f"\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
