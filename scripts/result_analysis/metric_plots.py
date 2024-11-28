import pandas as pd
import matplotlib.pyplot as plt

# Load the data from CSV
file_path = "/home/groups/dlmrimnd/jacob/data/combined_data/saved_results/run_proper_3/train_history.csv"
data = pd.read_csv(file_path)

# Directory to save the graphs
save_dir = "/home/groups/dlmrimnd/jacob/data/graphs/metrics/"

# Plotting function with save option and larger fonts
def plot_and_save_metric(metric_name, val_metric_name, title, ylabel, file_name):
    plt.figure(figsize=(10, 6))
    plt.plot(data['epoch'], data[metric_name], label='Training ' + ylabel, marker='o')
    plt.plot(data['epoch'], data[val_metric_name], label='Validation ' + ylabel, marker='o')
    
    # Adjusting font sizes for title, axis labels, and legend
    plt.title(title, fontsize=18)          # Larger title font
    plt.xlabel('Epoch', fontsize=16)       # Larger x-axis label font
    plt.ylabel(ylabel, fontsize=16)        # Larger y-axis label font
    plt.legend(fontsize=14)                # Larger legend font
    plt.grid(True)
    
    # Save the figure
    save_path = save_dir + file_name
    plt.savefig(save_path)
    plt.close()  # Close the plot to avoid displaying it

# Save plots to the specified directory with larger fonts
plot_and_save_metric('loss', 'val_loss', 'Training vs Validation Loss', 'Loss', 'training_vs_validation_loss.png')
plot_and_save_metric('movedSegm_sigmoid_sftDC', 'val_movedSegm_sigmoid_sftDC', 'Training vs Validation Moved Segmentation Sigmoid SftDC', 'Sigmoid SftDC', 'moved_segm_sigmoid_sftdc.png')
plot_and_save_metric('nonr_def_mean_absolute_error', 'val_nonr_def_mean_absolute_error', 'Training vs Validation Non-rigid Deformation Mean Absolute Error', 'Mean Absolute Error', 'non_rigid_def_mae.png')
plot_and_save_metric('srcSegm_sigmoid_DC', 'val_srcSegm_sigmoid_DC', 'Training vs Validation Source Segmentation Sigmoid Dice Coefficient', 'Sigmoid Dice Coefficient', 'source_segm_sigmoid_dice.png')
plot_and_save_metric('warpedSrc_MeanSquaredError', 'val_warpedSrc_MeanSquaredError', 'Training vs Validation Warped Source Mean Squared Error', 'Mean Squared Error', 'warped_source_mse.png')

# Plot and save the Dice coefficient (average of two Dice metrics) with larger fonts
data['dice_coefficient'] = (data['movedSegm_sigmoid_sftDC'] + data['srcSegm_sigmoid_DC']) / 2
data['val_dice_coefficient'] = (data['val_movedSegm_sigmoid_sftDC'] + data['val_srcSegm_sigmoid_DC']) / 2
plot_and_save_metric('dice_coefficient', 'val_dice_coefficient', 'Training vs Validation Dice Coefficient', 'Dice Coefficient', 'dice_coefficient.png')

# New function to plot the four loss metrics together using the correct column names and larger fonts
def plot_and_save_losses():
    plt.figure(figsize=(10, 6))
    
    # Corrected column names from your dataset
    plt.plot(data['epoch'], data['warpedSrc_loss'], label='L_reg', marker='o')   # Transformed Source Image Loss
    plt.plot(data['epoch'], data['movedSegm_loss'], label='L_com', marker='o')   # Transformed Segmentation Loss
    plt.plot(data['epoch'], data['srcSegm_loss'], label='L_seg', marker='o')     # Source Segmentation Loss
    plt.plot(data['epoch'], data['nonr_def_loss'], label='L_def', marker='o')    # Displacement Field Loss
    
    # Adjusting font sizes for title, axis labels, and legend
    plt.title('Losses Over Time', fontsize=18)    # Larger title font
    plt.xlabel('Epoch', fontsize=16)              # Larger x-axis label font
    plt.ylabel('Loss', fontsize=16)               # Larger y-axis label font
    plt.legend(fontsize=14)                       # Larger legend font
    plt.grid(True)
    
    # Save the figure
    save_path = save_dir + 'losses_over_time.png'
    plt.savefig(save_path)
    plt.close()

# Call the function to save the combined loss plot
plot_and_save_losses()
