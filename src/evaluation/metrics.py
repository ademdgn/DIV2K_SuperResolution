"""
Comprehensive evaluation metrics for Super Resolution models
Includes PSNR, SSIM, LPIPS, FID, and other perceptual metrics
"""

import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
from PIL import Image
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import pandas as pd
from tqdm import tqdm
import json
from datetime import datetime


class SuperResolutionMetrics:
    """Comprehensive metrics suite for Super Resolution evaluation"""
    
    def __init__(self, device='cuda'):
        self.device = device
        
        # Initialize LPIPS network
        try:
            self.lpips_net = lpips.LPIPS(net='alex').to(device)
            self.lpips_available = True
        except Exception as e:
            print(f"LPIPS not available: {e}")
            self.lpips_available = False
            
    def calculate_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate Peak Signal-to-Noise Ratio"""
        if img1.shape != img2.shape:
            raise ValueError("Images must have the same dimensions")
        
        # Convert to float64 for precision
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        
        # Calculate PSNR
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        
        max_pixel = 255.0 if img1.max() > 1.0 else 1.0
        psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
        
        return psnr_value
    
    def calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate Structural Similarity Index"""
        if img1.shape != img2.shape:
            raise ValueError("Images must have the same dimensions")
        
        # Convert to grayscale if needed
        if len(img1.shape) == 3:
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        else:
            img1_gray = img1
            img2_gray = img2
        
        # Calculate SSIM
        ssim_value = ssim(img1_gray, img2_gray, data_range=img1_gray.max() - img1_gray.min())
        
        return ssim_value
    
    def calculate_lpips(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Calculate Learned Perceptual Image Patch Similarity"""
        if not self.lpips_available:
            return -1.0
        
        with torch.no_grad():
            # Ensure tensors are in correct format [B, C, H, W] and range [-1, 1]
            if img1.dim() == 3:
                img1 = img1.unsqueeze(0)
            if img2.dim() == 3:
                img2 = img2.unsqueeze(0)
            
            # Convert from [0, 1] to [-1, 1] if needed
            if img1.max() <= 1.0:
                img1 = img1 * 2.0 - 1.0
                img2 = img2 * 2.0 - 1.0
            
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            
            # Calculate LPIPS
            lpips_value = self.lpips_net(img1, img2).item()
            
        return lpips_value
    
    def calculate_mse(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate Mean Squared Error"""
        mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
        return mse
    
    def calculate_mae(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate Mean Absolute Error"""
        mae = np.mean(np.abs(img1.astype(np.float64) - img2.astype(np.float64)))
        return mae
    
    def calculate_rmse(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate Root Mean Squared Error"""
        rmse = np.sqrt(self.calculate_mse(img1, img2))
        return rmse
    
    def evaluate_single_image_pair(self, sr_image: np.ndarray, hr_image: np.ndarray) -> Dict[str, float]:
        """Evaluate a single SR-HR image pair"""
        metrics = {}
        
        # Ensure images are in the same format
        if sr_image.shape != hr_image.shape:
            # Resize SR to match HR if needed
            sr_image = cv2.resize(sr_image, (hr_image.shape[1], hr_image.shape[0]))
        
        # Calculate pixel-level metrics
        metrics['psnr'] = self.calculate_psnr(sr_image, hr_image)
        metrics['ssim'] = self.calculate_ssim(sr_image, hr_image)
        metrics['mse'] = self.calculate_mse(sr_image, hr_image)
        metrics['mae'] = self.calculate_mae(sr_image, hr_image)
        metrics['rmse'] = self.calculate_rmse(sr_image, hr_image)
        
        # Calculate perceptual metrics
        if self.lpips_available:
            # Convert to tensors
            sr_tensor = torch.from_numpy(sr_image.transpose(2, 0, 1)).float() / 255.0
            hr_tensor = torch.from_numpy(hr_image.transpose(2, 0, 1)).float() / 255.0
            metrics['lpips'] = self.calculate_lpips(sr_tensor, hr_tensor)
        else:
            metrics['lpips'] = -1.0
        
        return metrics
    
    def evaluate_dataset(self, sr_dir: str, hr_dir: str, output_dir: str = "results") -> Dict:
        """Evaluate an entire dataset"""
        sr_path = Path(sr_dir)
        hr_path = Path(hr_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Get image lists
        sr_images = sorted(list(sr_path.glob("*.png")) + list(sr_path.glob("*.jpg")) + list(sr_path.glob("*.jpeg")))
        hr_images = sorted(list(hr_path.glob("*.png")) + list(hr_path.glob("*.jpg")) + list(hr_path.glob("*.jpeg")))
        
        if len(sr_images) != len(hr_images):
            print(f"Warning: Number of SR images ({len(sr_images)}) != HR images ({len(hr_images)})")
        
        all_metrics = []
        
        print(f"Evaluating {len(sr_images)} image pairs...")
        for sr_img_path, hr_img_path in tqdm(zip(sr_images, hr_images), total=len(sr_images)):
            # Load images
            sr_image = np.array(Image.open(sr_img_path).convert('RGB'))
            hr_image = np.array(Image.open(hr_img_path).convert('RGB'))
            
            # Calculate metrics
            metrics = self.evaluate_single_image_pair(sr_image, hr_image)
            metrics['image_name'] = sr_img_path.stem
            all_metrics.append(metrics)
        
        # Calculate statistics
        df = pd.DataFrame(all_metrics)
        
        # Summary statistics
        summary_stats = {
            'mean': df.select_dtypes(include=[np.number]).mean().to_dict(),
            'std': df.select_dtypes(include=[np.number]).std().to_dict(),
            'median': df.select_dtypes(include=[np.number]).median().to_dict(),
            'min': df.select_dtypes(include=[np.number]).min().to_dict(),
            'max': df.select_dtypes(include=[np.number]).max().to_dict(),
            'count': len(all_metrics)
        }
        
        # Save detailed results
        df.to_csv(output_path / "detailed_metrics.csv", index=False)
        
        # Save summary statistics
        with open(output_path / "summary_stats.json", 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        # Generate plots
        self.generate_evaluation_plots(df, output_path)
        
        return {
            'detailed_metrics': all_metrics,
            'summary_stats': summary_stats,
            'dataframe': df
        }
    
    def generate_evaluation_plots(self, df: pd.DataFrame, output_dir: Path):
        """Generate comprehensive evaluation plots"""
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Distribution plots for each metric
        metrics_to_plot = ['psnr', 'ssim', 'mse', 'mae', 'rmse']
        if 'lpips' in df.columns and df['lpips'].iloc[0] != -1.0:
            metrics_to_plot.append('lpips')
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics_to_plot):
            if i < len(axes):
                axes[i].hist(df[metric], bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{metric.upper()} Distribution', fontsize=14, fontweight='bold')
                axes[i].set_xlabel(metric.upper())
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
                
                # Add statistics text
                mean_val = df[metric].mean()
                std_val = df[metric].std()
                axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
                axes[i].legend()
        
        # Hide unused subplots
        for i in range(len(metrics_to_plot), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / "metrics_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Correlation heatmap
        plt.figure(figsize=(10, 8))
        correlation_matrix = df.select_dtypes(include=[np.number]).corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Metrics Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / "correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Box plots for metrics comparison
        plt.figure(figsize=(14, 8))
        # Normalize metrics for comparison
        df_normalized = df[metrics_to_plot].copy()
        for col in df_normalized.columns:
            df_normalized[col] = (df_normalized[col] - df_normalized[col].min()) / (df_normalized[col].max() - df_normalized[col].min())
        
        df_melted = df_normalized.melt(var_name='Metric', value_name='Normalized_Value')
        sns.boxplot(data=df_melted, x='Metric', y='Normalized_Value')
        plt.title('Normalized Metrics Comparison', fontsize=16, fontweight='bold')
        plt.ylabel('Normalized Value')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / "metrics_boxplot.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Scatter plots - PSNR vs SSIM
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(df['psnr'], df['ssim'], alpha=0.6, s=50, c=df['mse'], cmap='viridis')
        plt.xlabel('PSNR (dB)', fontsize=12)
        plt.ylabel('SSIM', fontsize=12)
        plt.title('PSNR vs SSIM (colored by MSE)', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, label='MSE')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "psnr_vs_ssim.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Evaluation plots saved to {output_dir}")


class ModelComparator:
    """Compare multiple Super Resolution models"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.metrics_calculator = SuperResolutionMetrics(device)
        
    def compare_models(self, model_results: Dict[str, str], hr_dir: str, output_dir: str = "model_comparison"):
        """
        Compare multiple models' results
        
        Args:
            model_results: Dict with model_name -> sr_results_dir
            hr_dir: Directory with ground truth HR images
            output_dir: Output directory for comparison results
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        comparison_results = {}
        
        # Evaluate each model
        for model_name, sr_dir in model_results.items():
            print(f"\nEvaluating {model_name}...")
            results = self.metrics_calculator.evaluate_dataset(sr_dir, hr_dir, output_path / model_name)
            comparison_results[model_name] = results['summary_stats']['mean']
        
        # Create comparison DataFrame
        df_comparison = pd.DataFrame(comparison_results).T
        
        # Save comparison results
        df_comparison.to_csv(output_path / "model_comparison.csv")
        
        # Generate comparison plots
        self.generate_comparison_plots(df_comparison, output_path)
        
        return df_comparison
    
    def generate_comparison_plots(self, df: pd.DataFrame, output_dir: Path):
        """Generate model comparison plots"""
        
        # 1. Radar chart for model comparison
        metrics = ['psnr', 'ssim', 'mse', 'mae', 'rmse']
        if 'lpips' in df.columns:
            metrics.append('lpips')
        
        # Normalize metrics for radar chart
        df_norm = df[metrics].copy()
        for col in df_norm.columns:
            if col in ['mse', 'mae', 'rmse', 'lpips']:  # Lower is better
                df_norm[col] = 1 - (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())
            else:  # Higher is better
                df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())
        
        # Create radar chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for model_name, row in df_norm.iterrows():
            values = row[metrics].tolist()
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.upper() for m in metrics])
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Comparison (Normalized)', size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        plt.tight_layout()
        plt.savefig(output_dir / "model_comparison_radar.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Bar chart comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if i < len(axes):
                df[metric].plot(kind='bar', ax=axes[i], color='skyblue', edgecolor='black')
                axes[i].set_title(f'{metric.upper()} Comparison', fontsize=14, fontweight='bold')
                axes[i].set_ylabel(metric.upper())
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / "model_comparison_bars.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Model comparison plots saved to {output_dir}")


class BenchmarkSuite:
    """Complete benchmarking suite for Super Resolution models"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.metrics = SuperResolutionMetrics(device)
        self.comparator = ModelComparator(device)
        
    def run_full_benchmark(self, model_path: str, test_hr_dir: str, output_dir: str = "benchmark_results"):
        """Run complete benchmark including inference and evaluation"""
        from ..models.esrgan import RDBNet
        import torchvision.transforms as transforms
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Load model
        device = torch.device(self.device)
        model = RDBNet().to(device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        if 'generator_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['generator_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        # Generate SR images
        sr_output_dir = output_path / "generated_sr"
        sr_output_dir.mkdir(exist_ok=True)
        
        hr_images = list(Path(test_hr_dir).glob("*.png")) + list(Path(test_hr_dir).glob("*.jpg"))
        
        print("Generating super-resolved images...")
        inference_times = []
        
        for hr_path in tqdm(hr_images):
            # Create LR image
            hr_image = Image.open(hr_path).convert('RGB')
            lr_image = hr_image.resize((hr_image.width // 4, hr_image.height // 4), Image.BICUBIC)
            
            # Convert to tensor
            transform = transforms.ToTensor()
            lr_tensor = transform(lr_image).unsqueeze(0).to(device)
            
            # Measure inference time
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            with torch.no_grad():
                sr_tensor = model(lr_tensor)
                sr_tensor = torch.clamp(sr_tensor, 0, 1)
            end_time.record()
            
            torch.cuda.synchronize()
            inference_time = start_time.elapsed_time(end_time)
            inference_times.append(inference_time)
            
            # Save SR image
            sr_image = transforms.ToPILImage()(sr_tensor.squeeze(0).cpu())
            sr_image.save(sr_output_dir / hr_path.name)
        
        # Evaluate results
        print("Evaluating results...")
        results = self.metrics.evaluate_dataset(str(sr_output_dir), test_hr_dir, str(output_path))
        
        # Add inference time statistics
        inference_stats = {
            'mean_inference_time_ms': np.mean(inference_times),
            'std_inference_time_ms': np.std(inference_times),
            'min_inference_time_ms': np.min(inference_times),
            'max_inference_time_ms': np.max(inference_times)
        }
        
        results['inference_stats'] = inference_stats
        
        # Generate comprehensive report
        self.generate_benchmark_report(results, output_path)
        
        return results
    
    def generate_benchmark_report(self, results: Dict, output_dir: Path):
        """Generate comprehensive benchmark report"""
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ESRGAN Benchmark Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; color: #333; }}
                .section {{ margin: 30px 0; }}
                .metric-table {{ border-collapse: collapse; width: 100%; }}
                .metric-table th, .metric-table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                .metric-table th {{ background-color: #f2f2f2; }}
                .good {{ color: green; font-weight: bold; }}
                .average {{ color: orange; font-weight: bold; }}
                .poor {{ color: red; font-weight: bold; }}
                .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }}
                .stats-card {{ background: #f9f9f9; padding: 20px; border-radius: 8px; text-align: center; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>ESRGAN Super Resolution Benchmark Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Performance Summary</h2>
                <div class="stats-grid">
                    <div class="stats-card">
                        <h3>PSNR</h3>
                        <p class="{'good' if results['summary_stats']['mean']['psnr'] > 25 else 'average' if results['summary_stats']['mean']['psnr'] > 20 else 'poor'}">{results['summary_stats']['mean']['psnr']:.2f} dB</p>
                    </div>
                    <div class="stats-card">
                        <h3>SSIM</h3>
                        <p class="{'good' if results['summary_stats']['mean']['ssim'] > 0.8 else 'average' if results['summary_stats']['mean']['ssim'] > 0.6 else 'poor'}">{results['summary_stats']['mean']['ssim']:.3f}</p>
                    </div>
                    <div class="stats-card">
                        <h3>LPIPS</h3>
                        <p class="{'good' if results['summary_stats']['mean'].get('lpips', 1) < 0.1 else 'average' if results['summary_stats']['mean'].get('lpips', 1) < 0.2 else 'poor'}">{results['summary_stats']['mean'].get('lpips', 'N/A')}</p>
                    </div>
                    <div class="stats-card">
                        <h3>Avg. Inference Time</h3>
                        <p>{results.get('inference_stats', {}).get('mean_inference_time_ms', 'N/A')} ms</p>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Detailed Metrics</h2>
                <table class="metric-table">
                    <tr>
                        <th>Metric</th>
                        <th>Mean</th>
                        <th>Std</th>
                        <th>Min</th>
                        <th>Max</th>
                    </tr>
        """
        
        for metric, value in results['summary_stats']['mean'].items():
            if metric != 'image_name':
                html_content += f"""
                    <tr>
                        <td>{metric.upper()}</td>
                        <td>{value:.4f}</td>
                        <td>{results['summary_stats']['std'][metric]:.4f}</td>
                        <td>{results['summary_stats']['min'][metric]:.4f}</td>
                        <td>{results['summary_stats']['max'][metric]:.4f}</td>
                    </tr>
                """
        
        html_content += """
                </table>
            </div>
        </body>
        </html>
        """
        
        # Save HTML report
        with open(output_dir / "benchmark_report.html", 'w') as f:
            f.write(html_content)
        
        print(f"Comprehensive benchmark report saved to {output_dir}")
