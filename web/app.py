"""
Interactive Web Interface for ESRGAN Super Resolution
Built with Streamlit
"""

import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io
import time
import sys
import os
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.esrgan import RDBNet
from src.evaluation.metrics import SuperResolutionMetrics
from src.utils.data_utils import tensor_to_numpy


# Page configuration
st.set_page_config(
    page_title="ESRGAN Super Resolution",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stProgress .st-bo {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path):
    """Load and cache the model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = RDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4)
    model = model.to(device)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if 'generator_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['generator_state_dict'])
            epoch = checkpoint.get('epoch', 'unknown')
        else:
            model.load_state_dict(checkpoint)
            epoch = 'unknown'
        
        model.eval()
        return model, device, epoch
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None


def super_resolve_image(model, image, device):
    """Super resolve a single image"""
    transform = transforms.ToTensor()
    
    # Convert PIL to tensor
    lr_tensor = transform(image).unsqueeze(0).to(device)
    
    # Measure inference time
    start_time = time.time()
    
    with torch.no_grad():
        sr_tensor = model(lr_tensor)
        sr_tensor = torch.clamp(sr_tensor, 0, 1)
    
    inference_time = time.time() - start_time
    
    # Convert back to PIL
    sr_image = transforms.ToPILImage()(sr_tensor.squeeze(0).cpu())
    
    return sr_image, inference_time


def calculate_metrics(sr_image, hr_image=None):
    """Calculate metrics between SR and HR images"""
    if hr_image is None:
        return {}
    
    try:
        metrics_calc = SuperResolutionMetrics('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Convert to numpy arrays
        sr_array = np.array(sr_image)
        hr_array = np.array(hr_image)
        
        # Resize if needed
        if sr_array.shape != hr_array.shape:
            hr_image_resized = hr_image.resize(sr_image.size, Image.LANCZOS)
            hr_array = np.array(hr_image_resized)
        
        metrics = metrics_calc.evaluate_single_image_pair(sr_array, hr_array)
        return metrics
    except Exception as e:
        st.error(f"Error calculating metrics: {e}")
        return {}


def main():
    # Header
    st.markdown('<h1 class="main-header">🖼️ ESRGAN Super Resolution</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("🛠️ Settings")
    
    # Model selection
    model_path = st.sidebar.text_input("Model Path", "esrgan_epoch_80.pth")
    
    if not os.path.exists(model_path):
        st.sidebar.error("Model file not found!")
        st.error("Please provide a valid model path in the sidebar.")
        return
    
    # Load model
    with st.spinner("Loading model..."):
        model, device, epoch = load_model(model_path)
    
    if model is None:
        return
    
    st.sidebar.success(f"✅ Model loaded (Epoch: {epoch})")
    st.sidebar.info(f"Device: {device}")
    
    # Main interface tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Single Image", "📊 Batch Processing", "📈 Model Analysis", "ℹ️ About"])
    
    with tab1:
        st.header("Single Image Super Resolution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input")
            
            # Image upload
            uploaded_file = st.file_uploader(
                "Choose an image...", 
                type=['png', 'jpg', 'jpeg'],
                help="Upload a low-resolution image to enhance"
            )
            
            if uploaded_file is not None:
                # Load and display input image
                input_image = Image.open(uploaded_file).convert('RGB')
                st.image(input_image, caption="Input Image", use_column_width=True)
                
                # Display image info
                st.info(f"📏 Size: {input_image.size[0]} × {input_image.size[1]} pixels")
                
                # Super resolution button
                if st.button("🚀 Enhance Image", type="primary"):
                    with st.spinner("Processing..."):
                        progress_bar = st.progress(0)
                        
                        # Perform super resolution
                        progress_bar.progress(50)
                        sr_image, inference_time = super_resolve_image(model, input_image, device)
                        progress_bar.progress(100)
                        
                        # Store results in session state
                        st.session_state.sr_image = sr_image
                        st.session_state.inference_time = inference_time
                        st.session_state.input_image = input_image
        
        with col2:
            st.subheader("Output")
            
            if hasattr(st.session_state, 'sr_image'):
                # Display super resolved image
                st.image(st.session_state.sr_image, caption="Super Resolved Image", use_column_width=True)
                
                # Display metrics
                col2a, col2b = st.columns(2)
                with col2a:
                    st.metric("Output Size", f"{st.session_state.sr_image.size[0]} × {st.session_state.sr_image.size[1]}")
                with col2b:
                    st.metric("Inference Time", f"{st.session_state.inference_time:.2f}s")
                
                # Calculate scale factor
                scale_x = st.session_state.sr_image.size[0] / st.session_state.input_image.size[0]
                scale_y = st.session_state.sr_image.size[1] / st.session_state.input_image.size[1]
                st.metric("Scale Factor", f"{scale_x:.1f}x")
                
                # Download button
                buf = io.BytesIO()
                st.session_state.sr_image.save(buf, format='PNG')
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="📥 Download Enhanced Image",
                    data=byte_im,
                    file_name="enhanced_image.png",
                    mime="image/png"
                )
            else:
                st.info("Upload an image and click 'Enhance Image' to see results here.")
        
        # Comparison section
        if hasattr(st.session_state, 'sr_image'):
            st.markdown("---")
            st.subheader("📊 Comparison")
            
            # Optional HR reference upload
            hr_reference = st.file_uploader(
                "Upload HR reference for metrics (optional)", 
                type=['png', 'jpg', 'jpeg'],
                help="Upload a high-resolution reference image to calculate quality metrics"
            )
            
            if hr_reference:
                hr_image = Image.open(hr_reference).convert('RGB')
                
                # Calculate metrics
                with st.spinner("Calculating metrics..."):
                    metrics = calculate_metrics(st.session_state.sr_image, hr_image)
                
                if metrics:
                    # Display metrics in cards
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("PSNR", f"{metrics.get('psnr', 0):.2f} dB")
                    with col2:
                        st.metric("SSIM", f"{metrics.get('ssim', 0):.3f}")
                    with col3:
                        st.metric("MSE", f"{metrics.get('mse', 0):.2f}")
                    with col4:
                        if metrics.get('lpips', -1) != -1:
                            st.metric("LPIPS", f"{metrics.get('lpips', 0):.3f}")
                        else:
                            st.metric("LPIPS", "N/A")
                
                # Side-by-side comparison
                st.subheader("Visual Comparison")
                comp_col1, comp_col2, comp_col3 = st.columns(3)
                
                with comp_col1:
                    st.image(st.session_state.input_image, caption="Input (LR)", use_column_width=True)
                with comp_col2:
                    st.image(st.session_state.sr_image, caption="Output (SR)", use_column_width=True)
                with comp_col3:
                    st.image(hr_image, caption="Reference (HR)", use_column_width=True)
    
    with tab2:
        st.header("📊 Batch Processing")
        st.info("🚧 Feature in development")
        
        st.markdown("""
        **Planned Features:**
        - Upload multiple images at once
        - Process entire folders
        - Download results as ZIP
        - Batch metrics calculation
        - Progress tracking for large batches
        """)
    
    with tab3:
        st.header("📈 Model Analysis")
        
        # Model information
        st.subheader("Model Information")
        
        # Calculate model size
        param_count = sum(p.numel() for p in model.parameters())
        param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Parameters", f"{param_count:,}")
        with col2:
            st.metric("Model Size", f"{param_size:.1f} MB")
        with col3:
            st.metric("Epoch", str(epoch))
        
        # Architecture visualization
        st.subheader("Model Architecture")
        st.info("ESRGAN with Residual Dense Blocks (RDB)")
        
        # Create a simple architecture diagram
        arch_data = {
            'Layer': ['Input', 'First Conv', 'RRDB Blocks (x23)', 'Trunk Conv', 'Upsampling', 'HR Conv', 'Output Conv'],
            'Channels': [3, 64, 64, 64, 64, 64, 3],
            'Description': [
                'RGB Input Image',
                '3×3 Convolution',
                'Residual Dense Blocks',
                '3×3 Convolution',
                '2× Upsampling (×2)',
                '3×3 Convolution',
                'RGB Output Image'
            ]
        }
        
        arch_df = pd.DataFrame(arch_data)
        st.dataframe(arch_df, use_container_width=True)
        
        # Performance metrics visualization
        st.subheader("Performance Benchmarks")
        
        # Sample benchmark data (replace with actual data)
        benchmark_data = {
            'Metric': ['PSNR', 'SSIM', 'LPIPS', 'Inference Time'],
            'Value': [28.5, 0.85, 0.12, 45.2],
            'Unit': ['dB', '', '', 'ms'],
            'Benchmark': ['> 25', '> 0.8', '< 0.2', '< 100']
        }
        
        benchmark_df = pd.DataFrame(benchmark_data)
        
        # Create gauge charts for metrics
        col1, col2 = st.columns(2)
        
        with col1:
            # PSNR gauge
            fig_psnr = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = 28.5,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "PSNR (dB)"},
                delta = {'reference': 25},
                gauge = {
                    'axis': {'range': [None, 35]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 20], 'color': "lightgray"},
                        {'range': [20, 25], 'color': "yellow"},
                        {'range': [25, 35], 'color': "green"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 30}}))
            st.plotly_chart(fig_psnr, use_container_width=True)
        
        with col2:
            # SSIM gauge
            fig_ssim = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = 0.85,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "SSIM"},
                delta = {'reference': 0.8},
                gauge = {
                    'axis': {'range': [None, 1]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 0.6], 'color': "lightgray"},
                        {'range': [0.6, 0.8], 'color': "yellow"},
                        {'range': [0.8, 1], 'color': "green"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.9}}))
            st.plotly_chart(fig_ssim, use_container_width=True)
    
    with tab4:
        st.header("ℹ️ About ESRGAN")
        
        st.markdown("""
        ## Enhanced Super-Resolution Generative Adversarial Networks (ESRGAN)
        
        ESRGAN is a state-of-the-art super-resolution model that uses deep learning to enhance image resolution
        while preserving fine details and textures.
        
        ### Key Features:
        - **4x Super Resolution**: Enhances images to 4 times their original resolution
        - **Perceptual Quality**: Optimized for visual quality using perceptual loss
        - **Real-time Processing**: Fast inference suitable for interactive applications
        - **Robust Architecture**: Uses Residual Dense Blocks for better feature extraction
        
        ### Technical Details:
        - **Generator**: RDBNet with 23 Residual Dense Blocks
        - **Discriminator**: Deep CNN with relativistic loss
        - **Training Dataset**: DIV2K high-quality images
        - **Scale Factor**: 4x upsampling
        
        ### Applications:
        - Photo enhancement and restoration
        - Medical image processing
        - Satellite image analysis
        - Digital art and content creation
        
        ### Model Performance:
        - **PSNR**: Measures pixel-level accuracy
        - **SSIM**: Structural similarity index
        - **LPIPS**: Perceptual similarity (lower is better)
        - **Inference Speed**: Optimized for real-time use
        
        ### Usage Tips:
        1. **Input Size**: Works best with images 64×64 to 512×512 pixels
        2. **Image Quality**: Clean, well-lit images give better results
        3. **File Format**: PNG recommended for best quality preservation
        4. **Processing Time**: Larger images take longer to process
        
        ---
        
        **Model Information:**
        - Framework: PyTorch
        - Architecture: ESRGAN
        - Training Data: DIV2K Dataset
        - Scale Factor: 4x
        
        **Citation:**
        ```
        Wang, X., Yu, K., Wu, S., Gu, J., Liu, Y., Dong, C., ... & Change Loy, C. (2018).
        ESRGAN: Enhanced super-resolution generative adversarial networks.
        In Proceedings of the European conference on computer vision (ECCV) workshops.
        ```
        """)


if __name__ == "__main__":
    main()
