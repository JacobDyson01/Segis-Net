import os

# Root directory where your output_files are located
root_dir = '/home/groups/dlmrimnd/akshit/output_files/'

# Walk through the root directory to find the mri subfolder
for subdir in os.listdir(root_dir):
    session_dir = os.path.join(root_dir, subdir)
    
    # Check if the session_dir is indeed a directory
    if os.path.isdir(session_dir):
        # Count the number of sessions
        session_count = len([s for s in os.listdir(root_dir) if s.startswith(subdir.split('_')[0])])
        
        # Process only if the patient has at least two sessions
        if session_count >= 2:
            mri_dir = os.path.join(session_dir, 'mri')
            
            mgz_file_path = os.path.join(mri_dir, 'aparc.DKTatlas+aseg.deep.withCC.mgz')
                
            if not os.path.exists(mgz_file_path):
                print(subdir)