import pandas as pd
import os

# File path to the output LaTeX file
output_file_path = '/home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/latex_table_output.tex'

# Check if the file already exists and delete it
if os.path.exists(output_file_path):
    os.remove(output_file_path)
    print(f"Existing file '{output_file_path}' has been deleted.")

# Load the CSV file (replace with your actual file path)
df = pd.read_csv('/home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/review_342008_20241001103952.csv')

# Function to create LaTeX table row
def create_latex_row(row):
    study_entry = f"\\cite{{{row['Covidence #']}}}"
    study_objective = row['Study Objectives'] if pd.notna(row['Study Objectives']) else "Other"
    roi = row['Region Of Interest'] if pd.notna(row['Region Of Interest']) else "Other"
    framework = row['Segmentation Framework'] if pd.notna(row['Segmentation Framework']) else "Other"
    validation = row['Validation'] if pd.notna(row['Validation']) else "Custom"
    modality = row['Modality'] if pd.notna(row['Modality']) else "Other"
    
    # Create a LaTeX formatted row with the updated column order
    latex_row = f"{study_entry} & {study_objective} & {roi} & {framework} & {validation} & {modality} \\\\\n\\hline\n"
    return latex_row

# Start creating LaTeX table structure for first table (24 rows)
latex_output_1 = """
\\begin{landscape}
\\begin{table}[H]
\\centering
\\scalebox{0.6}{
\\begin{tabular}{|c|c|c|c|c|c|}
\\hline
\\textbf{Study Entry} & \\textbf{Study Objectives} & \\textbf{ROI} & \\textbf{Segmentation Framework} & \\textbf{Validation} & \\textbf{Modality} \\\\
\\hline
"""

# Iterate over the first 24 rows in the dataframe
for index, row in df.head(24).iterrows():
    latex_row = create_latex_row(row)
    latex_output_1 += latex_row

# End the first table
latex_output_1 += """
\\end{tabular}
}
\\caption{Systematic Review Key Findings Part 1}
\\label{table:study_info_part1}
\\end{table}
\\end{landscape}
"""

# Start creating LaTeX table structure for second table (remaining rows)
latex_output_2 = """
\\begin{landscape}
\\begin{table}[H]
\\centering
\\scalebox{0.6}{
\\begin{tabular}{|c|c|c|c|c|c|}
\\hline
\\textbf{Study Entry} & \\textbf{Study Objectives} & \\textbf{ROI} & \\textbf{Segmentation Framework} & \\textbf{Validation} & \\textbf{Modality} \\\\
\\hline
"""

# Iterate over the remaining rows in the dataframe
for index, row in df.tail(24).iterrows():
    latex_row = create_latex_row(row)
    latex_output_2 += latex_row

# End the second table
latex_output_2 += """
\\end{tabular}
}
\\caption{Systematic Review Key Findings Part 2}
\\label{table:study_info_part2}
\\end{table}
\\end{landscape}
"""

# Save the LaTeX code to the specified path
with open(output_file_path, 'w') as f:
    f.write(latex_output_1)
    f.write(latex_output_2)

print(f"LaTeX tables have been successfully written to '{output_file_path}'.")
