import os
import torch
from PIL import Image
import torchvision.transforms as T
from improved_consistency_model_conditional import ConsistencySamplingAndEditing
from llvip.script import UNet
import numpy as np
import argparse
from torchvision.transforms.functional import resize
import math

def load_model(checkpoint_path):
    # Load the model
    model = UNet.from_pretrained(checkpoint_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    return model

def process_patches(image_tensor, model, consistency_sampling, device, patch_size=256, overlap=32):
    """Process image in patches to handle high resolution."""
    b, c, h, w = image_tensor.shape
    
    # Calculate steps and padding
    h_stride = patch_size - overlap
    w_stride = patch_size - overlap
    h_steps = math.ceil((h - overlap) / h_stride)
    w_steps = math.ceil((w - overlap) / w_stride)
    
    # Pad image if needed
    h_pad = h_steps * h_stride + overlap - h
    w_pad = w_steps * w_stride + overlap - w
    if h_pad > 0 or w_pad > 0:
        image_tensor = torch.nn.functional.pad(image_tensor, (0, w_pad, 0, h_pad), mode='reflect')
    
    # Storage for output
    output = torch.zeros_like(image_tensor)
    weight = torch.zeros_like(image_tensor)
    
    # Define the noise sigma schedule
    sigmas = [80.0, 40.0, 20.0, 10.0, 5.0, 2.5, 1.25, 0.625, 0.3125, 0.15625, 0.078125]
    
    # Process each patch
    for i in range(h_steps):
        for j in range(w_steps):
            # Extract patch
            h_start = i * h_stride
            w_start = j * w_stride
            patch = image_tensor[:, :, h_start:h_start + patch_size, w_start:w_start + patch_size]
            
            # Generate noise for patch
            noise = torch.randn_like(patch) * sigmas[0]
            
            # Process patch
            with torch.no_grad():
                processed_patch = consistency_sampling(
                    model=model,
                    y=noise,
                    v=patch,
                    sigmas=sigmas,
                    start_from_y=True,
                    add_initial_noise=False,
                    clip_denoised=True,
                    verbose=False,
                )
            
            # Create weight mask for blending (on the same device as the image)
            mask = torch.ones_like(patch)
            if overlap > 0:
                for axis in [2, 3]:  # H, W axes
                    tmp = torch.linspace(0, 1, overlap, device=device)
                    if axis == 2:
                        mask[:, :, :overlap] *= tmp[None, None, :, None]
                        mask[:, :, -overlap:] *= tmp.flip(0)[None, None, :, None]
                    else:
                        mask[:, :, :, :overlap] *= tmp[None, None, None, :]
                        mask[:, :, :, -overlap:] *= tmp.flip(0)[None, None, None, :]
            
            # Add processed patch to output
            output[:, :, h_start:h_start + patch_size, w_start:w_start + patch_size] += processed_patch * mask
            weight[:, :, h_start:h_start + patch_size, w_start:w_start + patch_size] += mask
    
    # Blend patches
    output = output / (weight + 1e-8)
    
    # Remove padding if added
    if h_pad > 0 or w_pad > 0:
        output = output[:, :, :h, :w]
    
    return output

def process_image(image_path, model, device, output_path=None):
    print(f"Processing input image: {image_path}")
    
    # Load and check the input image
    visible_image = Image.open(image_path)
    print(f"Original image size: {visible_image.size}, mode: {visible_image.mode}")
    
    # Properly handle 16-bit grayscale image
    if visible_image.mode == 'I;16':
        # Convert to 16-bit array
        img_array = np.array(visible_image)
        # Normalize to 0-1 range preserving 16-bit precision
        img_array = img_array.astype(np.float32) / 65535.0
        # Keep as float32 array
        visible_image = img_array
    else:
        visible_image = np.array(visible_image).astype(np.float32) / 255.0
    
    # Save the input image for verification (convert to 8-bit for viewing)
    input_debug_path = "debug_input.png"
    Image.fromarray((visible_image * 255).astype(np.uint8)).save(input_debug_path)
    print(f"Saved debug input image to: {input_debug_path}")
    
    # Convert single channel to 3-channel
    if len(visible_image.shape) == 2:
        visible_image = np.stack([visible_image] * 3, axis=-1)
    
    # Convert to tensor and move to device
    visible_tensor = torch.from_numpy(visible_image).permute(2, 0, 1).unsqueeze(0)
    visible_tensor = (visible_tensor * 2) - 1  # Normalize to [-1, 1]
    visible_tensor = visible_tensor.to(device)
    
    print(f"Tensor shape: {visible_tensor.shape}")
    print(f"Tensor value range: min={visible_tensor.min():.3f}, max={visible_tensor.max():.3f}")
    
    # Create a consistency sampling instance
    consistency_sampling = ConsistencySamplingAndEditing()
    
    # Process image in patches
    generated_infrared_tensor = process_patches(visible_tensor, model, consistency_sampling, device)
    
    print(f"Generated tensor range: min={generated_infrared_tensor.min():.3f}, max={generated_infrared_tensor.max():.3f}")
    
    # Convert the generated tensor to image
    generated_infrared = ((generated_infrared_tensor.squeeze(0).cpu() + 1) / 2).clamp(0, 1)
    
    # Convert to grayscale by taking the mean of channels
    if generated_infrared.shape[0] == 3:
        generated_infrared = generated_infrared.mean(dim=0, keepdim=True)
    
    # Convert to high bit depth array
    generated_infrared = (generated_infrared.numpy().transpose(1, 2, 0) * 65535).astype(np.uint16)
    
    # Check the output array statistics
    print(f"Output array shape: {generated_infrared.shape}")
    print(f"Output value range: min={generated_infrared.min()}, max={generated_infrared.max()}")
    
    # Convert to 16-bit grayscale image
    generated_image = Image.fromarray(generated_infrared.squeeze(), 'I;16')
    
    # Save the result if output path is provided
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:  # If there's a directory path
            os.makedirs(output_dir, exist_ok=True)
            
        # Save full 16-bit output
        generated_image.save(output_path.replace('.png', '.tif'))
        print(f"Generated 16-bit image saved to: {output_path.replace('.png', '.tif')}")
        
        # Save 8-bit version for viewing
        debug_output = Image.fromarray((generated_infrared.squeeze() / 256).astype(np.uint8))
        debug_output = T.functional.autocontrast(debug_output)
        debug_output.save('debug_' + output_path)
        print(f"8-bit debug version saved as: debug_{output_path}")

    return generated_image

def main():
    parser = argparse.ArgumentParser(description='Generate infrared image from visible image using trained model')
    parser.add_argument('--input', type=str, required=True, help='Path to input visible image')
    parser.add_argument('--output', type=str, default=None, help='Path to save output infrared image')
    parser.add_argument('--checkpoint', type=str, default='llvip/checkpoints/llvip',
                      help='Path to model checkpoint directory')
    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint)
    print("Model loaded successfully")

    # Process image
    print("Processing image...")
    output_path = args.output or f"results_{os.path.basename(args.input)}"
    result = process_image(args.input, model, device, output_path)
    print("Processing complete!")

if __name__ == "__main__":
    main() 