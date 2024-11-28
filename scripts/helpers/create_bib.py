import pandas as pd

# Load the CSV file (replace with your actual file path)
df = pd.read_csv('/home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/review_342008_included_csv_20241001104805.csv')

# Function to fix the author field format
def format_authors(authors_str):
    # Replace semicolons (;) with "and" for BibTeX format
    authors_str = authors_str.replace(';', ' and')
    return authors_str

# Function to create BibTeX entry for each row
def create_bibtex_entry(row):
    # Covidence number as the citation key, remove the # symbol if it exists
    citation_key = str(row['Covidence #']).replace("#", "")
    
    # Start creating the BibTeX entry
    bibtex_entry = f"@article{{{citation_key},\n"
    
    # Format the author field
    authors = format_authors(row['Authors'])
    bibtex_entry += f"  author = {{{authors}}},\n"
    
    bibtex_entry += f"  title = {{{row['Title']}}},\n"
    bibtex_entry += f"  journal = {{{row['Journal']}}},\n"
    bibtex_entry += f"  volume = {{{row['Volume']}}},\n"
    bibtex_entry += f"  number = {{{row['Issue']}}},\n"
    
    # Combine the Published Year and Published Month columns for the year entry
    if pd.notna(row['Published Year']):
        bibtex_entry += f"  year = {{{row['Published Year']}}},\n"
    if pd.notna(row['Published Month']):
        bibtex_entry += f"  month = {{{row['Published Month']}}},\n"
    
    bibtex_entry += f"  doi = {{{row['DOI']}}},\n"
    bibtex_entry += f"}}\n\n"
    
    return bibtex_entry

# Create a file to save all the BibTeX entries
with open('/home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/generated_references.bib', 'w') as f:
    for index, row in df.iterrows():
        # Generate the BibTeX entry for each reference
        bibtex_entry = create_bibtex_entry(row)
        # Write the entry to the file
        f.write(bibtex_entry)

print("BibTeX entries have been successfully written to 'generated_references.bib'.")
