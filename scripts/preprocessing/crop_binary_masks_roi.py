import os
import nibabel as nib

def crop_image(input_path, output_path, crop_coords):
    # Load the image
    img = nib.load(input_path)
    img_data = img.get_fdata()

    # Crop the image using the specified coordinates
    x_min, x_max, y_min, y_max, z_min, z_max = crop_coords
    cropped_img_data = img_data[x_min:x_max, y_min:y_max, z_min:z_max]

    # Save the cropped image
    cropped_img = nib.Nifti1Image(cropped_img_data, img.affine)
    nib.save(cropped_img, output_path)

def crop_binary_mask(input_path, output_path, crop_coords):
    crop_image(input_path, output_path, crop_coords)

input_dir = "/home/groups/dlmrimnd/jacob/data/MND/deformation_fields_reversed"
output_mask_dir = "/home/groups/dlmrimnd/jacob/data/MND/deformation_fields_reversed_roi"
crop_coords = crop_coords = (12, 188, 80, 160, 60, 156)  # Define your crop coordinates here

# Iterate over the binary masks
for sub_dir in os.listdir(input_dir):
    sub_id = os.path.basename(sub_dir)
    print(f"Processing {sub_id}...")

    # Define paths to binary masks
    mask_ses1 = os.path.join(input_dir, sub_dir, "deformation_1Warp.nii.gz")
    mask_ses1_roi = os.path.join(output_mask_dir, sub_id, "deformation_1Warp_roi.nii.gz")

    # Create output directories
    os.makedirs(os.path.dirname(mask_ses1_roi), exist_ok=True)

    # Crop the binary masks
    crop_binary_mask(mask_ses1, mask_ses1_roi, crop_coords)

    print(f"{sub_id} binary mask processed and saved.")

print("All binary masks processed.")