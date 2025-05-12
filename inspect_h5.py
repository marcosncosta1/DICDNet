import h5py
import sys
import os

def print_h5_item(name, obj, indent=""):
    """Recursively prints the structure of an HDF5 file."""
    print(f"{indent}- {name}: <{obj.__class__.__name__}>")
    if isinstance(obj, h5py.Dataset):
        print(f"{indent}  Shape: {obj.shape}")
        print(f"{indent}  Dtype: {obj.dtype}")
    elif isinstance(obj, h5py.Group):
        for key, item in obj.items():
            print_h5_item(key, item, indent + "  ")

# Get the filename from the command line arguments
if len(sys.argv) < 2:
    print("Usage: python your_script_name.py <path_to_your_file.h5>")
    sys.exit(1)

h5_file_path = sys.argv[1]

if not os.path.exists(h5_file_path):
    print(f"Error: File not found at {h5_file_path}")
    sys.exit(1)

try:
    # Open the .h5 file in read mode
    with h5py.File(h5_file_path, 'r') as f:
        print(f"Contents of {h5_file_path}:")
        print("-" * 30)
        # Start the recursive printing from the root group
        for key, item in f.items():
            print_h5_item(key, item)
        print("-" * 30)

except Exception as e:
    print(f"An error occurred while trying to read the HDF5 file: {e}")