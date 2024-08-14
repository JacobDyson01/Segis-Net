import os

# Path to the binary_masks directory
binary_masks_dir = '/home/groups/dlmrimnd/jacob/data/binary_masks'

# Iterate over all items in the binary_masks directory
for item in os.listdir(binary_masks_dir):
    item_path = os.path.join(binary_masks_dir, item)
    
    # Check if the item is a directory and matches the pattern 'sub-xxx_ses_x'
    if os.path.isdir(item_path) and item.startswith('sub-') and '_ses_' in item:
        # Replace the last underscore with a hyphen
        new_item = item.rsplit('_', 1)[0] + '-' + item.rsplit('_', 1)[1]
        new_item_path = os.path.join(binary_masks_dir, new_item)
        
        # Rename the directory
        os.rename(item_path, new_item_path)
        print(f'Renamed {item_path} to {new_item_path}')

print('Renaming completed.')