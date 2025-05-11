import os
import os.path
import argparse
import numpy as np
import torch
import time
import nibabel as nib
from skimage.transform import resize
from skimage.filters import gaussian  # For blur XLI mode
import glob
import re
# Ensure the DICDNet class definition is accessible
# If dicdnet.py is in the same directory, this should work.
# Otherwise, adjust sys.path or copy the class definition here.
from dicdnet import DICDNet

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="DICDNet_Test_Clinical_Adapted")
# Model loading arguments (from original test_DICDNet.py)
parser.add_argument("--model_dir", type=str, default="models/DICDNet_latest.pt",
                    help='path to trained DICDNet model file')
parser.add_argument('--num_M', type=int, default=32, help='the number of feature maps M_n')  # DICDNet specific
parser.add_argument('--num_Q', type=int, default=32,
                    help='the number of channel concatenation for Q in ProxNet_x')  # This seems like an ACDNet param, DICDNet paper mentions ResNet for ProxNet. Check dicdnet.py if Q is used. Assuming it's related to ProxNet depth/width.
parser.add_argument('--T', type=int, default=3,
                    help='the number of ResBlocks in every ProxNet')  # DICDNet specific (depth of ProxNets)
parser.add_argument('--S', type=int, default=10, help='Stage number S')  # DICDNet specific
parser.add_argument('--etaM', type=float, default=1.0, help='stepsize for updating M')  # DICDNet specific
parser.add_argument('--etaX', type=float, default=5.0, help='stepsize for updating X')  # DICDNet specific
parser.add_argument('--batchSize', type=int, default=1, help='inference input batch size (should be 1)')

# Data path arguments (NEW)
parser.add_argument("--input_low_dir", type=str, required=True, help='Directory containing X_low .nii.gz files')
parser.add_argument("--output_dir", type=str, required=True, help='Directory to save generated output .npy files')

# Processing arguments (NEW)
parser.add_argument("--img_size", type=int, default=416, help='Target size for input slices (DICDNet uses 416x416)')
parser.add_argument("--window_min", type=int, default=-1000, help='Minimum HU value for windowing')
parser.add_argument("--window_max", type=int, default=1000, help='Maximum HU value for windowing')
parser.add_argument("--metal_threshold_hu", type=int, default=2500,
                    help='HU threshold to segment metal for the non-metal mask M')
parser.add_argument("--slice_axis", type=int, default=2,
                    help='Axis along which to extract 2D slices (usually 2 for axial)')
parser.add_argument("--save_format", type=str, default="npy", choices=["npy", "png"],
                    help='Format to save output slices (npy recommended for [-1,1] range)')
parser.add_argument("--xli_mode", type=str, default="copy", choices=["copy", "blur", "approx_radon"],
                    help='Method to generate XLI input')
parser.add_argument("--blur_sigma", type=float, default=1.5, help='Sigma for Gaussian blur if xli_mode is "blur"')

# GPU arguments (from original)
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')

opt = parser.parse_args()

# --- GPU Setup ---
if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {opt.gpu_id}")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU.")
else:
    device = torch.device("cpu")
    print("Using CPU.")


# --- Utility Functions ---
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"--- Created output directory: {path} ---")
    else:
        print(f"--- Output directory exists: {path} ---")


def preprocess_dicdnet_input(slice_data_hu, target_size, win_min, win_max):
    """Preprocesses a single HU slice for DICDNet input: HU Window -> Norm [0,1] -> Scale [0,255] -> Resize."""
    img_slice_windowed = np.clip(slice_data_hu, win_min, win_max)
    if win_max > win_min:
        img_slice_normalized_01 = (img_slice_windowed - win_min) / (win_max - win_min)
    else:
        img_slice_normalized_01 = np.zeros_like(img_slice_windowed)
    img_slice_scaled_0_255 = img_slice_normalized_01 * 255.0
    img_resized = resize(img_slice_scaled_0_255, (target_size, target_size), anti_aliasing=True, preserve_range=True)
    img_tensor = torch.from_numpy(img_resized).float().unsqueeze(0).unsqueeze(0)
    return img_tensor


def generate_non_metal_mask_m(slice_data_hu, metal_threshold_hu, target_size):
    """Generates a non-metal mask M (1 for non-metal, 0 for metal) and preprocesses it."""
    metal_region_mask = (slice_data_hu >= metal_threshold_hu).astype(np.float32)
    non_metal_mask_m = 1.0 - metal_region_mask
    mask_resized = resize(non_metal_mask_m, (target_size, target_size), order=0, anti_aliasing=False,
                          preserve_range=True)
    mask_resized = np.clip(mask_resized, 0, 1)
    mask_tensor = torch.from_numpy(mask_resized).float().unsqueeze(0).unsqueeze(0)
    return mask_tensor


def generate_xli_for_dicdnet(xma_tensor_0_255, mode="copy", sigma=1.5, raw_hu_slice=None, raw_metal_mask=None):
    """Generates the XLI tensor (scaled [0,255]) based on the chosen mode."""
    if mode == "copy":
        return xma_tensor_0_255.clone()
    elif mode == "blur":
        xma_np = xma_tensor_0_255.squeeze().cpu().numpy()
        xli_blurred_np = gaussian(xma_np, sigma=sigma)
        xli_tensor = torch.from_numpy(xli_blurred_np).float().unsqueeze(0).unsqueeze(0).to(xma_tensor_0_255.device)
        return xli_tensor
    elif mode == "approx_radon":
        if raw_hu_slice is None or raw_metal_mask is None:
            raise ValueError("raw_hu_slice and raw_metal_mask are required for approx_radon mode.")
        from skimage.transform import radon, iradon  # Import locally for this mode

        print("    Generating approximate XLI using Radon...")
        theta = np.linspace(0., 180., max(raw_hu_slice.shape), endpoint=False)
        norm_min, norm_max = -1024.0, 3071.0
        image_norm_radon = np.clip((raw_hu_slice - norm_min) / (norm_max - norm_min), 0, 1)

        sinogram_ma = radon(image_norm_radon, theta=theta, circle=False)
        sinogram_mask = radon(raw_metal_mask.astype(float), theta=theta, circle=False)
        corruption_mask = sinogram_mask > 1e-6

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
        reconstructed_li_hu = reconstructed_li_norm_radon * (norm_max - norm_min) + norm_min

        # Preprocess this HU XLI like Xma for the network
        xli_tensor_0_255 = preprocess_dicdnet_input(reconstructed_li_hu, xma_tensor_0_255.shape[-1],  # target_size
                                                    opt.window_min, opt.window_max)  # Use main script's window
        print("    Approximate XLI (Radon) generated and preprocessed.")
        return xli_tensor_0_255.to(xma_tensor_0_255.device)
    else:
        raise ValueError(f"Invalid xli_mode: {mode}")


def save_dicdnet_output(output_tensor_0_255, filename, save_format="npy"):
    """Saves the DICDNet output. Normalizes from [0,255] to [-1,1] before saving."""
    output_slice_0_255 = output_tensor_0_255.squeeze().cpu().numpy()
    output_slice_neg1_1 = (output_slice_0_255 / 255.0) * 2.0 - 1.0
    output_slice_neg1_1 = np.clip(output_slice_neg1_1, -1.0, 1.0)

    if save_format == "npy":
        np.save(filename, output_slice_neg1_1)
    elif save_format == "png":
        from PIL import Image  # Import locally
        output_scaled_uint8 = ((output_slice_neg1_1 + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(output_scaled_uint8)
        img.save(filename)


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

start_time_main = time.time()

# --- Main Function ---
def main():
    mkdir(opt.output_dir)

    print('Loading DICDNet model ...')
    # Ensure opt has all necessary attributes for DICDNet constructor
    model = DICDNet(opt).to(device)
    try:
        model.load_state_dict(torch.load(opt.model_dir, map_location=device))
        print(f"Model loaded successfully from {opt.model_dir}")
    except Exception as e:
        print(f"Error loading model from {opt.model_dir}: {e}")
        return
    model.eval()

    low_files = sorted(glob.glob(os.path.join(opt.input_low_dir, '*.nii.gz')))
    if not low_files:
        print(f"Error: No input .nii.gz files found in {opt.input_low_dir}");
        return
    print(f"Found {len(low_files)} input NIfTI volumes to process.")

    total_slices_processed = 0;
    total_time = 0

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

                M_tensor = generate_non_metal_mask_m(slice_low_hu_raw, opt.metal_threshold_hu, opt.img_size)
                M_tensor = M_tensor.to(device)

                if torch.mean(M_tensor) > 0.999:  # If >99.9% is non-metal (mask is mostly 1s)
                    continue

                Xma_tensor_0_255 = preprocess_dicdnet_input(slice_low_hu_raw, opt.img_size, opt.window_min,
                                                            opt.window_max)
                Xma_tensor_0_255 = Xma_tensor_0_255.to(device)

                # Prepare raw inputs for approx_radon if needed
                raw_metal_mask_for_radon = (slice_low_hu_raw >= opt.metal_threshold_hu).astype(
                    np.float32) if opt.xli_mode == "approx_radon" else None
                XLI_tensor_0_255 = generate_xli_for_dicdnet(
                    Xma_tensor_0_255,
                    mode=opt.xli_mode,
                    sigma=opt.blur_sigma,
                    raw_hu_slice=slice_low_hu_raw if opt.xli_mode == "approx_radon" else None,
                    raw_metal_mask=raw_metal_mask_for_radon
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
            print(f"  Finished volume in {vol_end_time - vol_start_time:.2f} seconds.")
            total_time += (vol_end_time - vol_start_time)
        except Exception as e:
            print(f"Error processing file {low_filepath}: {e}")

    # --- (Final summary printing - same as ACDNet adapted script) ---
    end_time_main = time.time()
    print("\n" + "=" * 30)
    print("Processing Complete.")
    print(f"Total slices processed: {total_slices_processed}")
    if total_slices_processed > 0:
        avg_time_slice = total_time / total_slices_processed
        print(f"Average inference time per slice: {avg_time_slice:.4f} seconds")
    print(f"Generated output files saved in: {opt.output_dir}")
    from datetime import timedelta  # Ensure timedelta is imported
    print(f"Total execution time: {timedelta(seconds=int(end_time_main - start_time_main))}")
    print("=" * 30)


if __name__ == "__main__":
    main()
