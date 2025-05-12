import numpy as np

# Load the stacked masks
try:
    all_masks_stack = np.load("/home/mcosta/PycharmProjects/DICDNet/DICDNet/data/test/testmask.npy") # Ensure this path is correct
except FileNotFoundError:
    print("Error: testmask.npy not found. Please check the path.")
    exit()


# all_masks_stack should have shape (height, width, num_masks), e.g., (512, 512, 10)
if all_masks_stack.ndim != 3 or all_masks_stack.shape[2] != 10:
    print(f"Unexpected shape for testmask.npy: {all_masks_stack.shape}")
    print("Expected (height, width, 10)")
    exit()

print("Pixel areas from testmask.npy:")
mask_pixel_areas_from_npy = []
for i in range(all_masks_stack.shape[2]): # Iterate through the 10 masks
    mask_slice = all_masks_stack[:, :, i]
    # Assuming metal pixels are > 0 (e.g., 1 for binary masks)
    # If your masks use different values, adjust the condition accordingly.
    pixel_area = np.sum(mask_slice > 0) # Or np.count_nonzero(mask_slice) if strictly binary 0 or 1
    mask_pixel_areas_from_npy.append(pixel_area)
    print(f"Mask index {i} in .npy file: {pixel_area} pixels")

# This list now holds the pixel areas in the order they appear in the .npy file's last dimension
# e.g., mask_pixel_areas_from_npy[0] is the area of the 0th mask in the stack.
# You'll use this list as your `ordered_mask_sizes` if you rely on the .npy file.