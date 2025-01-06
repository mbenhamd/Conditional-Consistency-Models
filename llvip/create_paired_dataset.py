import os
import numpy as np
import math
from PIL import Image
import bm3d
from pathlib import Path
import argparse
import torch
import torchvision.transforms as T
from torch.nn.functional import interpolate

def apply_log_adaptation(img):
    """Apply logarithmic adaptation to enhance image contrast."""
    min_log = 100
    full_scale = 360000
    width = img.shape[1]
    height = img.shape[0]
    sq_min = math.sqrt(min_log)
    sq_full = math.sqrt(full_scale)

    a = (2**16 - 1) / (2 * math.sqrt(min_log * full_scale) - min_log)
    b = 2 * (2**16 - 1) / (2 * sq_full - sq_min)
    c = -(2**16 - 1) * sq_min / (2 * sq_full - sq_min)

    img_int = (img * (360000 / (2**16 - 1))).astype(np.uint32)
    img_int = img_int.flatten()

    img_int[img_int < min_log] = a * img_int[img_int < min_log]
    img_int[img_int >= min_log] = b * np.sqrt(img_int[img_int >= min_log]) + c

    img_int = np.reshape(img_int, (height, width))
    return img_int.astype(np.uint16)

def random_crop(img, size=512):
    """Randomly crop image to specified size."""
    if img.shape[0] < size or img.shape[1] < size:
        # Pad if image is smaller than crop size
        pad_h = max(0, size - img.shape[0])
        pad_w = max(0, size - img.shape[1])
        img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')
    
    h, w = img.shape
    top = np.random.randint(0, h - size + 1)
    left = np.random.randint(0, w - size + 1)
    return img[top:top+size, left:left+size]

def create_training_pair(img, crop_size=512, num_crops=4):
    """Create multiple training pairs from single image."""
    pairs = []
    
    # Normalize to 0-1 for BM3D
    img_float = img.astype(np.float32) / 65535.0
    
    # Apply BM3D to full image
    denoised = bm3d.bm3d(img_float, sigma_psd=0.1)
    
    # Convert back to 16-bit
    denoised = (denoised * 65535).astype(np.uint16)
    
    # Apply log adaptation
    enhanced = apply_log_adaptation(denoised)
    
    # Create multiple random crops
    for _ in range(num_crops):
        crop_source = random_crop(img, crop_size)
        crop_target = random_crop(enhanced, crop_size)
        pairs.append((crop_source, crop_target))
    
    return pairs

def process_image(input_path, output_dir_source, output_dir_target, crop_size=512):
    """Process a single image to create source-target pairs."""
    # Read 16-bit image
    img = np.array(Image.open(input_path))
    
    if img.dtype != np.uint16:
        raise ValueError(f"Expected 16-bit image, got {img.dtype}")
    
    # Create training pairs
    pairs = create_training_pair(img, crop_size=crop_size)
    
    # Save pairs
    filename = Path(input_path).stem
    for idx, (source, target) in enumerate(pairs):
        # Save full 16-bit versions
        Image.fromarray(source).save(
            os.path.join(output_dir_source, f"{filename}_crop{idx}.tif")
        )
        Image.fromarray(target).save(
            os.path.join(output_dir_target, f"{filename}_crop{idx}.tif")
        )
        
        # Save 8-bit previews
        Image.fromarray((source / 256).astype(np.uint8)).save(
            os.path.join(output_dir_source, f"{filename}_crop{idx}_preview.png")
        )
        Image.fromarray((target / 256).astype(np.uint8)).save(
            os.path.join(output_dir_target, f"{filename}_crop{idx}_preview.png")
        )

def create_paired_dataset(input_dir, output_base_dir, crop_size=512):
    """Create paired dataset from directory of input images."""
    # Create output directories
    output_dir_source = os.path.join(output_base_dir, 'source')
    output_dir_target = os.path.join(output_base_dir, 'target')
    os.makedirs(output_dir_source, exist_ok=True)
    os.makedirs(output_dir_target, exist_ok=True)
    
    # Process all .tif files
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.tif', '.tiff')):
            input_path = os.path.join(input_dir, filename)
            try:
                print(f"Processing {filename}...")
                process_image(input_path, output_dir_source, output_dir_target, crop_size)
                print(f"Successfully processed {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Create paired dataset for CCM training')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, required=True, help='Base directory for output paired dataset')
    parser.add_argument('--crop_size', type=int, default=512, help='Size of random crops')
    args = parser.parse_args()
    
    create_paired_dataset(args.input_dir, args.output_dir, args.crop_size)
    print("Dataset creation complete!")

if __name__ == "__main__":
    main() 