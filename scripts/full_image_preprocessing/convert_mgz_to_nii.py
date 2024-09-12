import nibabel as nib
import sys

def convert_mgz_to_nii(input_mgz_path, output_nii_path):
    # Load the .mgz file
    mgz_image = nib.load(input_mgz_path)
    
    # Save it as a .nii.gz file
    nib.save(mgz_image, output_nii_path)
    print(f"Converted {input_mgz_path} to {output_nii_path}")

if __name__ == "__main__":
    # Read input and output file paths from the command line arguments
    input_mgz = sys.argv[1]
    output_nii = sys.argv[2]
    
    convert_mgz_to_nii(input_mgz, output_nii)
