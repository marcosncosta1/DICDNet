import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import argparse

def to_grayscale(image):
    """
    Convert an image to grayscale if it has more than one channel.
    For CT scans, we assume that taking the first channel is sufficient.
    """
    if image.ndim >= 3 and image.shape[-1] > 1:
        # If image has an alpha channel (RGBA) or is RGB, take only the first channel.
        return image[..., 0]
    return image

def dynamic_win_size(image, default=7):
    """
    Compute an appropriate window size for SSIM based on spatial dimensions.
    This function assumes that 'image' is already grayscale.
    """
    h, w = image.shape[:2]
    min_dim = min(h, w)
    if min_dim < 3:
        return None
    win = min(default, min_dim)
    if win % 2 == 0:
        win -= 1
    return win

def evaluate_dicdnet_image_and_hu(
    results_dir="./save_results",
    output_file="dicdnet_image_quality_metrics.txt"
):
    """
    Evaluate PSNR and SSIM for normal images and HU images produced by DICDNet,
    using the naming convention:
      - GT normal:  n_gt_image.png
      - Net normal: n_dicdnet_image.png
      - GT HU:      n_gt_hu.png
      - Net HU:     n_dicdnet_hu.png

    Directory structure is assumed to be:
      results_dir/
       ├─ gt/
       │   ├─ image/   (files like n_gt_image.png)
       │   └─ hu/      (files like n_gt_hu.png)
       └─ DICDNet/
           ├─ image/   (files like n_dicdnet_image.png)
           └─ hu/      (files like n_dicdnet_hu.png)
    """
    # Define subfolder paths
    gt_image_dir = os.path.join(results_dir, "gt", "image")
    net_image_dir = os.path.join(results_dir, "DICDNet", "image")
    gt_hu_dir = os.path.join(results_dir, "gt", "hu")
    net_hu_dir = os.path.join(results_dir, "DICDNet", "hu")

    # Check that these directories exist
    for d in [gt_image_dir, net_image_dir, gt_hu_dir, net_hu_dir]:
        if not os.path.isdir(d):
            print(f"Warning: directory does not exist => {d}")

    # Get list of GT normal image filenames matching pattern n_gt_image.png
    all_gt_filenames = sorted(
        f for f in os.listdir(gt_image_dir) if f.endswith("_gt_image.png")
    )

    psnr_img_vals, ssim_img_vals = [], []
    psnr_hu_vals, ssim_hu_vals = [], []
    table_rows = []
    table_header = (
        f"{'Filename':<25}{'PSNR(img)':>10}{'SSIM(img)':>10}"
        f"{'PSNR(hu)':>10}{'SSIM(hu)':>10}\n"
        + "-" * 65
    )

    for gt_filename in all_gt_filenames:
        # Extract prefix (e.g., "1000" from "1000_gt_image.png")
        prefix = gt_filename.replace("_gt_image.png", "")
        net_img_filename = f"{prefix}_dicdnet_image.png"
        gt_hu_filename = f"{prefix}_gt_hu.png"
        net_hu_filename = f"{prefix}_dicdnet_hu.png"

        gt_img_path = os.path.join(gt_image_dir, gt_filename)
        net_img_path = os.path.join(net_image_dir, net_img_filename)
        gt_hu_path = os.path.join(gt_hu_dir, gt_hu_filename)
        net_hu_path = os.path.join(net_hu_dir, net_hu_filename)

        # Check that the corresponding files exist
        if not os.path.exists(net_img_path):
            print(f"Skipping {gt_filename} - missing {net_img_path}")
            continue
        if not os.path.exists(gt_hu_path):
            print(f"Skipping {gt_filename} - missing {gt_hu_path}")
            continue
        if not os.path.exists(net_hu_path):
            print(f"Skipping {gt_filename} - missing {net_hu_path}")
            continue

        # Load images and convert to float64 for computation
        gt_img = np.squeeze(plt.imread(gt_img_path).astype(np.float64))
        net_img = np.squeeze(plt.imread(net_img_path).astype(np.float64))
        gt_hu_img = np.squeeze(plt.imread(gt_hu_path).astype(np.float64))
        net_hu_img = np.squeeze(plt.imread(net_hu_path).astype(np.float64))

        # Convert to grayscale if needed
        gt_img = to_grayscale(gt_img)
        net_img = to_grayscale(net_img)
        gt_hu_img = to_grayscale(gt_hu_img)
        net_hu_img = to_grayscale(net_hu_img)

        # Debug: print image ranges and shapes
        print(f"\nProcessing {gt_filename}:")
        print(f"GT normal range: [{gt_img.min():.3f}, {gt_img.max():.3f}], shape: {gt_img.shape}")
        print(f"Recon normal range: [{net_img.min():.3f}, {net_img.max():.3f}], shape: {net_img.shape}")
        print(f"GT HU range: [{gt_hu_img.min():.3f}, {gt_hu_img.max():.3f}], shape: {gt_hu_img.shape}")
        print(f"Recon HU range: [{net_hu_img.min():.3f}, {net_hu_img.max():.3f}], shape: {net_hu_img.shape}")

        # For normal images: compute dynamic window size based on spatial dimensions
        win_size_img = dynamic_win_size(gt_img, default=7)
        if win_size_img is None:
            print(f"Skipping {gt_filename} for normal SSIM due to small spatial dimensions: {gt_img.shape}")
            continue

        # Since these images are grayscale, set channel_axis=None
        data_range_img = max(gt_img.max(), net_img.max()) - min(gt_img.min(), net_img.min())
        if data_range_img <= 0:
            data_range_img = 1e-6

        psnr_img_val = psnr(gt_img, net_img, data_range=data_range_img)
        ssim_img_val = ssim(gt_img, net_img,
                            data_range=data_range_img,
                            win_size=win_size_img,
                            channel_axis=None)

        # For HU images: similarly compute dynamic window size
        win_size_hu = dynamic_win_size(gt_hu_img, default=7)
        if win_size_hu is None:
            print(f"Skipping {gt_filename} for HU SSIM due to small spatial dimensions: {gt_hu_img.shape}")
            continue

        data_range_hu = max(gt_hu_img.max(), net_hu_img.max()) - min(gt_hu_img.min(), net_hu_img.min())
        if data_range_hu <= 0:
            data_range_hu = 1e-6

        psnr_hu_val = psnr(gt_hu_img, net_hu_img, data_range=data_range_hu)
        ssim_hu_val = ssim(gt_hu_img, net_hu_img,
                           data_range=data_range_hu,
                           win_size=win_size_hu,
                           channel_axis=None)

        psnr_img_vals.append(psnr_img_val)
        ssim_img_vals.append(ssim_img_val)
        psnr_hu_vals.append(psnr_hu_val)
        ssim_hu_vals.append(ssim_hu_val)

        row = (f"{gt_filename:<25}{psnr_img_val:>10.2f}{ssim_img_val:>10.4f}"
               f"{psnr_hu_val:>10.2f}{ssim_hu_val:>10.4f}")
        table_rows.append(row)

    def mean_std_str(arr):
        if len(arr) == 0:
            return "0.00 ± 0.00"
        return f"{np.mean(arr):.2f} ± {np.std(arr):.2f}"

    summary = (
        "\nSummary (mean ± std)\n"
        f"PSNR(img): {mean_std_str(psnr_img_vals)}\n"
        f"SSIM(img): {np.mean(ssim_img_vals):.4f} ± {np.std(ssim_img_vals):.4f}\n"
        f"PSNR(hu):  {mean_std_str(psnr_hu_vals)}\n"
        f"SSIM(hu):  {np.mean(ssim_hu_vals):.4f} ± {np.std(ssim_hu_vals):.4f}\n"
    )

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    full_report = (
        f"DICDNet Image Quality Evaluation\n"
        f"Date: {timestamp}\n"
        f"Results Directory: {results_dir}\n\n"
        + table_header + "\n" +
        "\n".join(table_rows) +
        "\n" + summary
    )

    print(full_report)
    with open(output_file, 'w') as f:
        f.write(full_report)

    print(f"\nReport saved to: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DICDNet image quality metrics")
    parser.add_argument("--results_dir", type=str, default="./save_results", help="Directory containing DICDNet results")
    parser.add_argument("--output_file", type=str, default="dicdnet_image_quality_metrics.txt", help="Output filename for the report")
    args = parser.parse_args()

    evaluate_dicdnet_image_and_hu(args.results_dir, args.output_file)
