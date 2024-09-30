import pandas as pd
import matplotlib.pyplot as plt

# Load the data from CSV
file_path = "/home/groups/dlmrimnd/jacob/data/combined_data/saved_results/run1/train_history.csv"
data = pd.read_csv(file_path)

# Directory to save the graphs
save_dir = "/home/groups/dlmrimnd/jacob/data/graphs/"

# Plotting function with save option
def plot_and_save_metric(metric_name, val_metric_name, title, ylabel, file_name):
    plt.figure(figsize=(10, 6))
    plt.plot(data['epoch'], data[metric_name], label='Training ' + ylabel, marker='o')
    plt.plot(data['epoch'], data[val_metric_name], label='Validation ' + ylabel, marker='o')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    save_path = save_dir + file_name
    plt.savefig(save_path)
    plt.close()  # Close the plot to avoid displaying it

# Save plots to the specified directory
plot_and_save_metric('loss', 'val_loss', 'Training vs Validation Loss', 'Loss', 'training_vs_validation_loss.png')
plot_and_save_metric('movedSegm_sigmoid_sftDC', 'val_movedSegm_sigmoid_sftDC', 'Training vs Validation Moved Segmentation Sigmoid SftDC', 'Sigmoid SftDC', 'moved_segm_sigmoid_sftdc.png')
plot_and_save_metric('nonr_def_mean_absolute_error', 'val_nonr_def_mean_absolute_error', 'Training vs Validation Non-rigid Deformation Mean Absolute Error', 'Mean Absolute Error', 'non_rigid_def_mae.png')
plot_and_save_metric('srcSegm_sigmoid_DC', 'val_srcSegm_sigmoid_DC', 'Training vs Validation Source Segmentation Sigmoid Dice Coefficient', 'Sigmoid Dice Coefficient', 'source_segm_sigmoid_dice.png')
plot_and_save_metric('warpedSrc_MeanSquaredError', 'val_warpedSrc_MeanSquaredError', 'Training vs Validation Warped Source Mean Squared Error', 'Mean Squared Error', 'warped_source_mse.png')

# Add more metrics as needed following the same structure
