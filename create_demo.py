# Demo Image Creation Script

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

def create_demo_images():
    """Create demo images for README"""
    
    # Create demo directory
    os.makedirs('demo', exist_ok=True)
    
    # Low resolution input image (256x256)
    lr_size = (256, 256)
    lr_img = Image.new('RGB', lr_size, color='white')
    draw = ImageDraw.Draw(lr_img)
    
    # Draw some pattern
    for i in range(0, lr_size[0], 20):
        for j in range(0, lr_size[1], 20):
            color = (
                int(128 + 100 * np.sin(i/20) * np.cos(j/20)),
                int(128 + 100 * np.sin(i/15) * np.cos(j/15)), 
                int(128 + 100 * np.sin(i/25) * np.cos(j/25))
            )
            color = tuple(max(0, min(255, c)) for c in color)
            draw.rectangle([i, j, i+20, j+20], fill=color)
    
    # Add text
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    draw.text((10, 10), "Original\n256x256", fill='black', font=font)
    lr_img.save('demo/input_low_res.jpg', quality=85)
    
    # High resolution output image (1024x1024) - simulated
    hr_size = (1024, 1024)
    hr_img = lr_img.resize(hr_size, Image.LANCZOS)
    
    # Enhance the image (simulate super resolution effect)
    draw_hr = ImageDraw.Draw(hr_img)
    try:
        font_hr = ImageFont.truetype("arial.ttf", 48)
    except:
        font_hr = ImageFont.load_default()
    
    draw_hr.text((20, 20), "Enhanced 4x\n1024x1024", fill='red', font=font_hr)
    hr_img.save('demo/output_super_res.jpg', quality=95)
    
    # Create before/after comparison images
    # Before 1
    before1 = Image.new('RGB', (200, 200), color='lightblue')
    draw1 = ImageDraw.Draw(before1)
    draw1.text((10, 90), "Before\n200x200", fill='darkblue')
    before1.save('demo/before_1.jpg')
    
    # After 1
    after1 = before1.resize((800, 800), Image.LANCZOS)
    draw_a1 = ImageDraw.Draw(after1)
    draw_a1.text((20, 360), "After 4x\n800x800", fill='red', font=font_hr)
    after1.save('demo/after_1.jpg')
    
    # Before 2
    before2 = Image.new('RGB', (180, 180), color='lightgreen')
    draw2 = ImageDraw.Draw(before2)
    draw2.text((10, 80), "Before\n180x180", fill='darkgreen')
    before2.save('demo/before_2.jpg')
    
    # After 2
    after2 = before2.resize((720, 720), Image.LANCZOS)
    draw_a2 = ImageDraw.Draw(after2)
    draw_a2.text((20, 320), "After 4x\n720x720", fill='red', font=font_hr)
    after2.save('demo/after_2.jpg')
    
    print("Demo images created successfully!")
    print("Created files:")
    print("  demo/input_low_res.jpg")
    print("  demo/output_super_res.jpg") 
    print("  demo/before_1.jpg")
    print("  demo/after_1.jpg")
    print("  demo/before_2.jpg")
    print("  demo/after_2.jpg")

if __name__ == "__main__":
    create_demo_images()
