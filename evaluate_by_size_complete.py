import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
import argparse
import re


def to_grayscale(image):
    if image.ndim == 3 and image.shape[-1] > 1:
        return image[..., 0]
    elif image.ndim == 3 and image.shape[-1] == 1:
        return np.squeeze(image, axis=-1)
    return image


def dynamic_win_size(image, default=7):
    if image.ndim != 2:
        # print(f"Warning: dynamic_win_size expected 2D image, got shape {image.shape}")
        return None
    h, w = image.shape
    min_dim = min(h, w)
    if min_dim < 3:
        return None
    win = min(default, min_dim)
    if win % 2 == 0:
        win -= 1
    return win if win >= 3 else None


def get_mask_index_from_prefix(filename_prefix, num_masks_total=10):
    """
    Determines the mask index (0 to num_masks_total-1) based on the filename_prefix.
    Assumes filename_prefix is the 'n' from 'n_gt_image.png' and is 1-indexed.
    """
    try:
        # Assuming prefix is 'n' from 'n_gt_image.png' and n is 1-indexed.
        sequential_id = int(filename_prefix)
        if sequential_id <= 0:
            print(
                f"Warning: Sequential ID '{sequential_id}' from prefix '{filename_prefix}' is not positive. Cannot determine mask index.")
            return None
        mask_index = (sequential_id - 1) % num_masks_total  # (1-1)%10=0, (10-1)%10=9, (11-1)%10=0
        return mask_index
    except ValueError:
        # Fallback: Try to parse if there's a _maskX pattern, though less likely given user description
        match = re.search(r'_mask(\d+)', filename_prefix)
        if match:
            return int(match.group(1))
        print(f"Warning: Could not determine mask index from prefix '{filename_prefix}'. "
              "It's not a simple number for sequential ID, nor does it match '_maskX'.")
        return None


def evaluate_model_metrics_by_size(  # Renamed for clarity
        results_dir="./save_results",
        model_name="DICDNet",
        output_dir="."
):
    # --- Configuration based on your inspected mask sizes (from testmask.npy) ---
    # Index 0 of this list corresponds to mask '0.h5' (or mask_index 0)
    # Index 1 corresponds to mask '1.h5' (or mask_index 1), and so on.
    ordered_mask_sizes = [  # Corresponds to mask_index 0 through 9
        371, 1338, 688, 84, 171,
        54, 1329, 3119, 182, 180
    ]
    num_masks_total = len(ordered_mask_sizes)

    def get_mask_group(mask_pixel_area):
        # Using your specific mask pixel areas to define groups
        if mask_pixel_area in {54, 84}:
            return "G1"  # Your Smallest
        elif mask_pixel_area in {171, 180}:
            return "G2"
        elif mask_pixel_area in {182, 371}:
            return "G3"  # Your Medium
        elif mask_pixel_area in {688, 1329}:
            return "G4"
        elif mask_pixel_area in {1338, 3119}:
            return "G5"  # Your Largest
        else:
            print(f"Critical Warning: Mask area {mask_pixel_area} from 'ordered_mask_sizes' "
                  "does not fall into predefined groups in 'get_mask_group'. Check definitions.")
            return None

    # --- End Configuration ---

    gt_image_dir = os.path.join(results_dir, "gt", "image")
    net_image_dir = os.path.join(results_dir, model_name, "image")
    gt_hu_dir = os.path.join(results_dir, "gt", "hu")
    net_hu_dir = os.path.join(results_dir, model_name, "hu")

    for d_path in [gt_image_dir, net_image_dir, gt_hu_dir, net_hu_dir]:
        if not os.path.isdir(d_path):
            print(f"Error: Required directory does not exist => {d_path}")
            return

    all_gt_filenames_unsorted = [
        f for f in os.listdir(gt_image_dir) if f.endswith("_gt_image.png") and f.replace("_gt_image.png", "").isdigit()
    ]
    if not all_gt_filenames_unsorted:
        print(f"Error: No suitable '*_gt_image.png' files (with numeric prefixes) found in {gt_image_dir}")
        return

    # Sort filenames numerically based on the prefix 'n'
    all_gt_filenames = sorted(all_gt_filenames_unsorted, key=lambda x: int(x.replace("_gt_image.png", "")))

    metrics_by_group = {
        "G1": {"psnr_img": [], "ssim_img": [], "psnr_hu": [], "ssim_hu": [], "count": 0},
        "G2": {"psnr_img": [], "ssim_img": [], "psnr_hu": [], "ssim_hu": [], "count": 0},
        "G3": {"psnr_img": [], "ssim_img": [], "psnr_hu": [], "ssim_hu": [], "count": 0},
        "G4": {"psnr_img": [], "ssim_img": [], "psnr_hu": [], "ssim_hu": [], "count": 0},
        "G5": {"psnr_img": [], "ssim_img": [], "psnr_hu": [], "ssim_hu": [], "count": 0}
    }
    table_rows_individual = []
    individual_table_header = (
            f"{'Filename':<20}{'Prefix':<10}{'MaskIdx':>8}{'MaskArea':>10}{'Group':>8}"
            f"{'PSNR(img)':>12}{'SSIM(img)':>12}"
            f"{'PSNR(hu)':>12}{'SSIM(hu)':>12}\n"
            + "-" * 120  # Adjusted length
    )

    processed_files_count = 0
    skipped_due_to_mask_index = 0
    skipped_due_to_grouping = 0
    skipped_due_to_missing_files = 0
    skipped_due_to_loading_error = 0
    skipped_due_to_shape_mismatch = 0
    skipped_due_to_ssim_win_size = 0

    for gt_filename in all_gt_filenames:
        prefix = gt_filename.replace("_gt_image.png", "")

        # Construct network output filenames
        # Assuming the original script's naming for DICDNet, general for others
        if model_name == "DICDNet":
            net_img_filename = f"{prefix}_dicdnet_image.png"
            net_hu_filename = f"{prefix}_dicdnet_hu.png"
        else:
            # Generic naming for other models - you might need to adjust this
            # if your other models don't follow "prefix_modelname_image.png"
            net_img_filename = f"{prefix}_{model_name.lower()}_image.png"
            net_hu_filename = f"{prefix}_{model_name.lower()}_hu.png"

        gt_hu_filename = f"{prefix}_gt_hu.png"

        gt_img_path = os.path.join(gt_image_dir, gt_filename)
        net_img_path = os.path.join(net_image_dir, net_img_filename)
        gt_hu_path = os.path.join(gt_hu_dir, gt_hu_filename)
        net_hu_path = os.path.join(net_hu_dir, net_hu_filename)

        if not (os.path.exists(net_img_path) and \
                os.path.exists(gt_hu_path) and \
                os.path.exists(net_hu_path)):
            # print(f"Debug: Skipping {gt_filename} - one or more corresponding files missing.")
            # print(f"  Checked: {net_img_path}, {gt_hu_path}, {net_hu_path}")
            skipped_due_to_missing_files += 1
            continue

        mask_index = get_mask_index_from_prefix(prefix, num_masks_total)
        if mask_index is None:  # Error message already printed in function
            skipped_due_to_mask_index += 1
            continue

        if not (0 <= mask_index < len(ordered_mask_sizes)):
            print(
                f"Skipping {gt_filename} - determined mask_index {mask_index} is out of bounds for 'ordered_mask_sizes'.")
            skipped_due_to_mask_index += 1
            continue

        current_mask_pixel_area = ordered_mask_sizes[mask_index]
        group_key = get_mask_group(current_mask_pixel_area)

        if not group_key:
            # Error message already printed in function
            skipped_due_to_grouping += 1
            continue

        try:
            gt_img_raw = plt.imread(gt_img_path).astype(np.float64)
            net_img_raw = plt.imread(net_img_path).astype(np.float64)
            gt_hu_img_raw = plt.imread(gt_hu_path).astype(np.float64)
            net_hu_img_raw = plt.imread(net_hu_path).astype(np.float64)
        except Exception as e:
            print(f"Skipping {gt_filename} - error loading images: {e}")
            skipped_due_to_loading_error += 1
            continue

        gt_img = to_grayscale(np.squeeze(gt_img_raw))
        net_img = to_grayscale(np.squeeze(net_img_raw))
        gt_hu_img = to_grayscale(np.squeeze(gt_hu_img_raw))
        net_hu_img = to_grayscale(np.squeeze(net_hu_img_raw))

        if gt_img.shape != net_img.shape:
            # print(f"Skipping {gt_filename} - normal image shape mismatch: GT {gt_img.shape}, Net {net_img.shape}")
            skipped_due_to_shape_mismatch += 1
            continue
        if gt_hu_img.shape != net_hu_img.shape:
            # print(f"Skipping {gt_filename} - HU image shape mismatch: GT {gt_hu_img.shape}, Net {net_hu_img.shape}")
            skipped_due_to_shape_mismatch += 1
            continue

        # Calculate for Normal Images
        psnr_img_val, ssim_img_val = np.nan, np.nan
        data_range_psnr_img = gt_img.max() - gt_img.min() if gt_img.max() > gt_img.min() else 1.0
        if data_range_psnr_img == 0 and np.all(gt_img == net_img):
            psnr_img_val = np.inf  # Perfect match case
        elif data_range_psnr_img > 0:
            psnr_img_val = sk_psnr(gt_img, net_img, data_range=data_range_psnr_img)

        win_size_img = dynamic_win_size(gt_img)
        if win_size_img is None:
            # print(f"Skipping SSIM for {gt_filename} (normal) - small image: {gt_img.shape}")
            skipped_due_to_ssim_win_size += 1
        else:
            data_range_ssim_img = gt_img.max() - gt_img.min()  # For SSIM, scikit-image recommends actual range of GT
            data_range_ssim_img = data_range_ssim_img if data_range_ssim_img > 0 else 1.0
            try:
                ssim_img_val = sk_ssim(gt_img, net_img, data_range=data_range_ssim_img, win_size=win_size_img,
                                       channel_axis=None, gaussian_weights=True, use_sample_covariance=False, K1=0.01,
                                       K2=0.03)
            except ValueError as e:  # Sometimes happens if win_size is too large for a very flat image despite checks
                print(
                    f"SSIM calculation error for {gt_filename} (normal): {e}. Win size: {win_size_img}, Shape: {gt_img.shape}")

        # Calculate for HU Images
        psnr_hu_val, ssim_hu_val = np.nan, np.nan
        data_range_psnr_hu = gt_hu_img.max() - gt_hu_img.min() if gt_hu_img.max() > gt_hu_img.min() else 1.0
        if data_range_psnr_hu == 0 and np.all(gt_hu_img == net_hu_img):
            psnr_hu_val = np.inf
        elif data_range_psnr_hu > 0:
            psnr_hu_val = sk_psnr(gt_hu_img, net_hu_img, data_range=data_range_psnr_hu)

        win_size_hu = dynamic_win_size(gt_hu_img)
        if win_size_hu is None:
            # print(f"Skipping SSIM for {gt_filename} (HU) - small image: {gt_hu_img.shape}")
            skipped_due_to_ssim_win_size += 1
        else:
            data_range_ssim_hu = gt_hu_img.max() - gt_hu_img.min()
            data_range_ssim_hu = data_range_ssim_hu if data_range_ssim_hu > 0 else 1.0
            try:
                ssim_hu_val = sk_ssim(gt_hu_img, net_hu_img, data_range=data_range_ssim_hu, win_size=win_size_hu,
                                      channel_axis=None, gaussian_weights=True, use_sample_covariance=False, K1=0.01,
                                      K2=0.03)
            except ValueError as e:
                print(
                    f"SSIM calculation error for {gt_filename} (HU): {e}. Win size: {win_size_hu}, Shape: {gt_hu_img.shape}")

        if not (np.isnan(psnr_img_val) or np.isnan(ssim_img_val) or np.isnan(psnr_hu_val) or np.isnan(ssim_hu_val)):
            metrics_by_group[group_key]["psnr_img"].append(psnr_img_val)
            metrics_by_group[group_key]["ssim_img"].append(ssim_img_val)
            metrics_by_group[group_key]["psnr_hu"].append(psnr_hu_val)
            metrics_by_group[group_key]["ssim_hu"].append(ssim_hu_val)
            metrics_by_group[group_key]["count"] += 1
            processed_files_count += 1

        row_individual = (f"{gt_filename:<20}{prefix:<10}{mask_index:>8}{current_mask_pixel_area:>10}{group_key:>8}"
                          f"{psnr_img_val:>12.2f}{ssim_img_val:>12.4f}"
                          f"{psnr_hu_val:>12.2f}{ssim_hu_val:>12.4f}")
        table_rows_individual.append(row_individual)

    report_lines = []
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    report_lines.append(f"{model_name} Image Quality Evaluation by Metal Size")
    report_lines.append(f"Date: {timestamp}")
    report_lines.append(f"Results Directory: {os.path.abspath(results_dir)}")
    report_lines.append(f"Model Evaluated: {model_name}")
    report_lines.append(f"Total GT files found: {len(all_gt_filenames_unsorted)}")
    report_lines.append(f"Total files processed and metrics collected: {processed_files_count}")
    if skipped_due_to_missing_files > 0: report_lines.append(
        f"  Skipped due to missing corresponding files: {skipped_due_to_missing_files}")
    if skipped_due_to_mask_index > 0: report_lines.append(
        f"  Skipped due to mask index error: {skipped_due_to_mask_index}")
    if skipped_due_to_grouping > 0: report_lines.append(
        f"  Skipped due to mask area not in group: {skipped_due_to_grouping}")
    if skipped_due_to_loading_error > 0: report_lines.append(
        f"  Skipped due to image loading error: {skipped_due_to_loading_error}")
    if skipped_due_to_shape_mismatch > 0: report_lines.append(
        f"  Skipped due to shape mismatch: {skipped_due_to_shape_mismatch}")
    if skipped_due_to_ssim_win_size > 0: report_lines.append(
        f"  SSIM skipped (some images) due to small dimensions: {skipped_due_to_ssim_win_size} instances")

    report_lines.append("\n--- Metrics by Metal Size Group (mean ± std) ---")
    header_line1 = f"{'Group Label':<28}{'Count':>7}{'PSNR(img)':>20}{'SSIM(img)':>22}{'PSNR(hu)':>20}{'SSIM(hu)':>22}"
    header_line2 = f"{'-' * 28}{'-' * 7:>7}{'-' * 20:>20}{'-' * 22:>22}{'-' * 20:>20}{'-' * 22:>22}"
    report_lines.append(header_line1)
    report_lines.append(header_line2)

    group_display_config = [  # From largest impact to smallest
        ("G5", "Large       ({1338, 3119}px)"),
        ("G4", "Large-Med   ({688, 1329}px)"),
        ("G3", "Medium      ({182, 371}px)"),
        ("G2", "Med-Small   ({171, 180}px)"),
        ("G1", "Small       ({54, 84}px)")
    ]
    all_psnr_img, all_ssim_img, all_psnr_hu, all_ssim_hu = [], [], [], []

    for group_key_internal, group_label_display in group_display_config:
        data = metrics_by_group[group_key_internal]
        count = data["count"]

        all_psnr_img.extend(data["psnr_img"])  # These will be non-NaN due to earlier check
        all_ssim_img.extend(data["ssim_img"])
        all_psnr_hu.extend(data["psnr_hu"])
        all_ssim_hu.extend(data["ssim_hu"])

        if count == 0:
            line = f"{group_label_display:<28}{count:>7}{'N/A':>20}{'N/A':>22}{'N/A':>20}{'N/A':>22}"
        else:
            psnr_img_m, psnr_img_s = np.mean(data["psnr_img"]), np.std(data["psnr_img"])
            ssim_img_m, ssim_img_s = np.mean(data["ssim_img"]), np.std(data["ssim_img"])
            psnr_hu_m, psnr_hu_s = np.mean(data["psnr_hu"]), np.std(data["psnr_hu"])
            ssim_hu_m, ssim_hu_s = np.mean(data["ssim_hu"]), np.std(data["ssim_hu"])
            line = (f"{group_label_display:<28}{count:>7}"
                    f"{psnr_img_m:>12.2f} ± {psnr_img_s:<5.2f}"
                    f"{ssim_img_m:>14.4f} ± {ssim_img_s:<5.4f}"
                    f"{psnr_hu_m:>12.2f} ± {psnr_hu_s:<5.2f}"
                    f"{ssim_hu_m:>14.4f} ± {ssim_hu_s:<5.4f}")
        report_lines.append(line)
    report_lines.append(header_line2)

    report_lines.append("\n--- Overall Summary (mean ± std across all processed files) ---")
    if processed_files_count > 0:
        report_lines.append(
            f"PSNR(img): {np.mean(all_psnr_img):.2f} ± {np.std(all_psnr_img):.2f} (from {len(all_psnr_img)} images)")
        report_lines.append(f"SSIM(img): {np.mean(all_ssim_img):.4f} ± {np.std(all_ssim_img):.4f}")
        report_lines.append(f"PSNR(hu):  {np.mean(all_psnr_hu):.2f} ± {np.std(all_psnr_hu):.2f}")
        report_lines.append(f"SSIM(hu):  {np.mean(all_ssim_hu):.4f} ± {np.std(all_ssim_hu):.4f}")
    else:
        report_lines.append("No data successfully processed for overall summary.")

    report_lines.append("\n--- Individual File Metrics Log ---")
    report_lines.append(individual_table_header)
    report_lines.extend(table_rows_individual)

    final_report_str = "\n".join(report_lines)
    print(final_report_str)

    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"{model_name.lower()}_metrics_by_size.txt")

    with open(output_filename, 'w') as f:
        f.write(final_report_str)
    print(f"\nDetailed report saved to: {os.path.abspath(output_filename)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Model image quality metrics by metal size.")
    parser.add_argument("--results_dir", type=str, default="./save_results",
                        help="Base directory containing 'gt' and model output subdirectories (e.g., DICDNet/, ACDNet/).")
    parser.add_argument("--model_name", type=str, default="DICDNet",
                        help="Name of the model subdirectory. This is also used for output filenames if not 'DICDNet'.")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory where the metrics report file will be saved.")
    args = parser.parse_args()

    evaluate_model_metrics_by_size(
        results_dir=args.results_dir,
        model_name=args.model_name,
        output_dir=args.output_dir
    )