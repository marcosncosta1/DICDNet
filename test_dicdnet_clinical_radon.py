import os
import os.path
import argparse
import numpy as np
import torch
import time
import nibabel as nib
from skimage.transform import resize, radon, iradon  # Added radon, iradon
from skimage.filters import gaussian  # For blur XLI mode
import glob
import re
from dicdnet import DICDNet

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="DICDNet_Test_Clinical_RadonLI")
# Model loading arguments
parser.add_argument("--model_dir", type=str, default="models/DICDNet_latest.pt",
                    help='path to trained DICDNet model file')
parser.add_argument('--num_M', type=int, default=32, help='the number of feature maps')
parser.add_argument('--num_Q', type=int, default=32, help='the number of channel concatenation')
parser.add_argument('--T', type=int, default=3, help='the number of ResBlocks in every ProxNet')
parser.add_argument('--S', type=int, default=10, help='Stage number')
parser.add_argument('--etaM', type=float, default=1.0, help='stepsize for updating M')
parser.add_argument('--etaX', type=float, default=5.0, help='stepsize for updating X')
parser.add_argument('--batchSize', type=int, default=1, help='inference input batch size')
# Data path arguments
parser.add_argument("--input_low_dir", type=str, required=True, help='Directory containing X_low .nii.gz files')
parser.add_argument("--output_dir", type=str, required=True, help='Directory to save generated output .npy files')
# Processing arguments
parser.add_argument("--img_size", type=int, default=416, help='Target size for input slices')
parser.add_argument("--window_min", type=int, default=-1000, help='Minimum HU value for windowing')
parser.add_argument("--window_max", type=int, default=1000, help='Maximum HU value for windowing')
parser.add_argument("--metal_threshold_hu", type=int, default=2500,
                    help='HU threshold for non-metal mask M & for Radon LI')
parser.add_argument("--slice_axis", type=int, default=2, help='Axis for slicing')
parser.add_argument("--save_format", type=str, default="npy", choices=["npy", "png"], help='Save format')
parser.add_argument("--xli_mode", type=str, default="approx_radon", choices=["copy", "blur", "approx_radon"],
                    help='XLI generation mode')  # Defaulting to approx_radon
parser.add_argument("--blur_sigma", type=float, default=1.5, help='Sigma for blur mode')
# GPU arguments
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
opt = parser.parse_args()

# --- GPU Setup ---
if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu"); print("CUDA not available, using CPU.")
else:
    device = torch.device("cpu"); print("Using CPU.")
if str(device) == "cuda": print(f"Using GPU: {opt.gpu_id}")


# --- Utility Functions ---
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path); print(f"--- Created output directory: {path} ---")
    else:
        print(f"--- Output directory exists: {path} ---")


def preprocess_dicdnet_input(slice_data_hu, target_size, win_min, win_max):
    img_slice_windowed = np.clip(slice_data_hu, win_min, win_max)
    if win_max > win_min:
        img_slice_normalized_01 = (img_slice_windowed - win_min) / (win_max - win_min)
    else:
        img_slice_normalized_01 = np.zeros_like(img_slice_windowed)
    img_slice_scaled_0_255 = img_slice_normalized_01 * 255.0
    img_resized = resize(img_slice_scaled_0_255, (target_size, target_size), anti_aliasing=True, preserve_range=True)
    return torch.from_numpy(img_resized).float().unsqueeze(0).unsqueeze(0)


def generate_non_metal_mask_m(slice_data_hu, metal_threshold_hu, target_size):
    metal_region_mask = (slice_data_hu >= metal_threshold_hu).astype(np.float32)
    non_metal_mask_m = 1.0 - metal_region_mask
    mask_resized = resize(non_metal_mask_m, (target_size, target_size), order=0, anti_aliasing=False,
                          preserve_range=True)
    mask_resized = np.clip(mask_resized, 0, 1)
    return torch.from_numpy(mask_resized).float().unsqueeze(0).unsqueeze(0)


def generate_approx_radon_li_hu(xma_slice_hu, metal_region_mask_binary, theta=None):
    """ Generates approximate LI in HU space using Radon transforms. """
    print("    Generating approximate XLI using Radon...")
    if theta is None: theta = np.linspace(0., 180., max(xma_slice_hu.shape), endpoint=False)

    norm_min_radon, norm_max_radon = -1024.0, 3071.0  # For Radon stability
    image_norm_radon = np.clip((xma_slice_hu - norm_min_radon) / (norm_max_radon - norm_min_radon), 0, 1)

    sinogram_ma = radon(image_norm_radon, theta=theta, circle=False)
    sinogram_metal_mask = radon(metal_region_mask_binary.astype(float), theta=theta, circle=False)
    corruption_mask = sinogram_metal_mask > 1e-6

    sinogram_li = sinogram_ma.copy()
    height, width = sinogram_li.shape
    for i in range(width):
        col = sinogram_li[:, i];
        mask_col = corruption_mask[:, i]
        if np.any(mask_col):
            valid_indices = np.where(~mask_col)[0];
            invalid_indices = np.where(mask_col)[0]
            if len(valid_indices) > 1:
                interp_values = np.interp(invalid_indices, valid_indices, col[valid_indices])
                col[invalid_indices] = interp_values
            elif len(valid_indices) == 1:
                col[invalid_indices] = col[valid_indices[0]]
            else:
                col[invalid_indices] = 0.0

    reconstructed_li_norm_radon = iradon(sinogram_li, theta=theta, circle=False, filter_name='ramp')
    reconstructed_li_hu = reconstructed_li_norm_radon * (norm_max_radon - norm_min_radon) + norm_min_radon
    print("    Approximate XLI (Radon) generated in HU space.")
    return reconstructed_li_hu


def generate_xli_for_dicdnet(xma_tensor_0_255, mode="copy", sigma=1.5,
                             raw_hu_slice_for_radon=None, raw_metal_mask_for_radon=None,
                             target_size_for_radon_output=416, win_min_for_radon_output=-1000,
                             win_max_for_radon_output=1000):
    if mode == "copy":
        return xma_tensor_0_255.clone()
    elif mode == "blur":
        xma_np = xma_tensor_0_255.squeeze().cpu().numpy()
        xli_blurred_np = gaussian(xma_np, sigma=sigma)
        return torch.from_numpy(xli_blurred_np).float().unsqueeze(0).unsqueeze(0).to(xma_tensor_0_255.device)
    elif mode == "approx_radon":
        if raw_hu_slice_for_radon is None or raw_metal_mask_for_radon is None:
            raise ValueError("raw_hu_slice and raw_metal_mask are required for approx_radon mode.")

        xli_hu_approx = generate_approx_radon_li_hu(raw_hu_slice_for_radon, raw_metal_mask_for_radon)
        # Now preprocess this HU XLI to match the [0,255] scaled Xma format
        xli_tensor_0_255 = preprocess_dicdnet_input(xli_hu_approx, target_size_for_radon_output,
                                                    win_min_for_radon_output, win_max_for_radon_output)
        return xli_tensor_0_255.to(xma_tensor_0_255.device)
    else:
        raise ValueError(f"Invalid xli_mode: {mode}")


def save_dicdnet_output(output_tensor_0_255, filename, save_format="npy"):
    output_slice_0_255 = output_tensor_0_255.squeeze().cpu().numpy()
    output_slice_neg1_1 = (output_slice_0_255 / 255.0) * 2.0 - 1.0
    output_slice_neg1_1 = np.clip(output_slice_neg1_1, -1.0, 1.0)
    if save_format == "npy":
        np.save(filename, output_slice_neg1_1)
    elif save_format == "png":
        from PIL import Image
        output_scaled_uint8 = ((output_slice_neg1_1 + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(output_scaled_uint8).save(filename)


def load_nifti_slice_raw(filepath, slice_index, axis=2):
    try:
        nii_img = nib.load(filepath);
        data = nii_img.get_fdata()
        if axis == 0:
            slice_data = data[slice_index, :, :]
        elif axis == 1:
            slice_data = data[:, slice_index, :]
        else:
            slice_data = data[:, :, slice_index]
        return slice_data
    except Exception as e:
        print(f"Error loading NIfTI {filepath} slice {slice_index}: {e}"); return None


# --- Main Function ---
def main():
    # (Initialize start_time_main as in previous corrected version)
    start_time_main = time.time()
    mkdir(opt.output_dir)

    print('Loading DICDNet model ...')
    model = DICDNet(opt).to(device)
    try:
        model.load_state_dict(torch.load(opt.model_dir, map_location=device))
        print(f"Model loaded successfully from {opt.model_dir}")
    except Exception as e:
        print(f"Error loading model from {opt.model_dir}: {e}"); return
    model.eval()

    low_files = sorted(glob.glob(os.path.join(opt.input_low_dir, '*.nii.gz')))
    if not low_files: print(f"Error: No input .nii.gz files found in {opt.input_low_dir}"); return
    print(f"Found {len(low_files)} input NIfTI volumes to process.")

    total_slices_processed = 0;
    total_script_time = 0

    for i, low_filepath in enumerate(low_files):
        base_filename = os.path.basename(low_filepath).replace('.nii.gz', '')
        print(f"\nProcessing file {i + 1}/{len(low_files)}: {base_filename}")

        try:
            data_low_full_volume = nib.load(low_filepath).get_fdata()
            num_slices = data_low_full_volume.shape[opt.slice_axis]
            print(f"  Volume shape: {data_low_full_volume.shape}, Slices: {num_slices}")
            vol_start_time = time.time()

            for slice_idx in range(num_slices):
                slice_low_hu_raw = load_nifti_slice_raw(low_filepath, slice_idx, opt.slice_axis)
                if slice_low_hu_raw is None: continue

                # Generate Non-Metal Mask (M for DICDNet) from raw HU slice
                M_tensor = generate_non_metal_mask_m(slice_low_hu_raw, opt.metal_threshold_hu, opt.img_size)
                M_tensor = M_tensor.to(device)

                # For Radon XLI, we need the binary metal region mask (not the non-metal M)
                metal_region_mask_binary_for_radon = (slice_low_hu_raw >= opt.metal_threshold_hu).astype(np.float32)
                if torch.mean(M_tensor) > 0.999 and np.sum(
                        metal_region_mask_binary_for_radon) == 0:  # If M is all non-metal AND no metal for radon
                    continue

                # Preprocess Xma (the input metal-corrupted image)
                Xma_tensor_0_255 = preprocess_dicdnet_input(slice_low_hu_raw, opt.img_size, opt.window_min,
                                                            opt.window_max)
                Xma_tensor_0_255 = Xma_tensor_0_255.to(device)

                # Generate XLI
                XLI_tensor_0_255 = generate_xli_for_dicdnet(
                    Xma_tensor_0_255,
                    mode=opt.xli_mode,
                    sigma=opt.blur_sigma,
                    raw_hu_slice_for_radon=slice_low_hu_raw if opt.xli_mode == "approx_radon" else None,
                    raw_metal_mask_for_radon=metal_region_mask_binary_for_radon if opt.xli_mode == "approx_radon" else None,
                    target_size_for_radon_output=opt.img_size,
                    win_min_for_radon_output=opt.window_min,
                    win_max_for_radon_output=opt.window_max
                )

                with torch.no_grad():
                    _, ListX, _ = model(Xma_tensor_0_255, XLI_tensor_0_255, M_tensor)
                output_tensor_0_255 = ListX[-1]

                output_filename_base = f"{base_filename}_slice{slice_idx:04d}_dicdnet_out"
                output_filepath = os.path.join(opt.output_dir, f"{output_filename_base}.{opt.save_format}")
                save_dicdnet_output(output_tensor_0_255, output_filepath, opt.save_format)

                total_slices_processed += 1
                if (slice_idx + 1) % 100 == 0:
                    print(f"    Processed slice {slice_idx + 1}/{num_slices}")

            vol_end_time = time.time()
            current_vol_time = vol_end_time - vol_start_time
            print(f"  Finished volume in {current_vol_time:.2f} seconds.")
            total_script_time += current_vol_time
        except Exception as e:
            print(f"Error processing file {low_filepath}: {e}")

    end_time_main = time.time()
    print("\n" + "=" * 30)
    print("Processing Complete.")
    print(f"Total slices processed: {total_slices_processed}")
    if total_slices_processed > 0:
        avg_time_slice = total_script_time / total_slices_processed
        print(f"Average inference time per slice (based on volume processing): {avg_time_slice:.4f} seconds")
    print(f"Generated output files saved in: {opt.output_dir}")
    from datetime import timedelta
    print(f"Total script execution time: {timedelta(seconds=int(end_time_main - start_time_main))}")
    print("=" * 30)


if __name__ == "__main__":
    main()
