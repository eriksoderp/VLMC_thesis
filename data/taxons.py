import re
import requests
from collections import defaultdict
import random

# Tar in en .nwk (från TimeTree) och spottar ut en .txt med alla taxons
def extract_taxons(file_path, output_file):

    with open(file_path, "r") as file:
        newick_string = file.read()

    taxon_pattern = re.compile(r'(?<=\(|,)[^():,;\s]+')
    taxon_names = taxon_pattern.findall(newick_string)

    taxon_names = [taxon.replace('_', ' ') for taxon in taxon_names]

    with open(output_file, "w") as file:
        for taxon in taxon_names:
            file.write(taxon + "\n")

### Fixar TimeTree output
def fix_file(file_path, fix_file_path):
    
    with open(file_path, 'r') as f:
        eukaryotes = f.read().splitlines()
    with open(fix_file_path, 'r') as f:
        fixes = f.read().splitlines()

    fix_dict = {}
    for fix in fixes:
        key, value = fix.split(' (replaced with ')
        key = key.strip()
        value = value[:-1]
        fix_dict[key] = value

    # Apply the fixes to the eukaryotes list
    for i in range(len(eukaryotes)):
        if eukaryotes[i] in fix_dict:
            eukaryotes[i] = fix_dict[eukaryotes[i]]

    # Write the updated eukaryotes list to a new file
    with open('prokaryotes_taxons2_updated.txt', 'w') as f:
        f.write('\n'.join(eukaryotes))

def remove_spaces(input_file, output_file):
    with open(input_file, 'r') as infile:
        text = infile.read()

    words = text.split()
    output = []

    for word in words:
        if not word.strip():
            continue

        if output and word[0].isupper():
            output.append('\n' + word)
        else:
            output.append(word)

    with open(output_file, 'w') as outfile:
        outfile.write(' '.join(output))

### GET KINGDOM
def get_kingdom(taxon):
    url = f"https://api.gbif.org/v1/species/match?name={taxon}&rank=species&verbose=true"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        kingdom = data.get('kingdom')
        return kingdom
    return None


def process_text_file(input_file, output_file):
    with open(input_file, 'r') as infile:
        taxons = infile.readlines()

    kingdom_taxons = defaultdict(list)
    output = []

    for taxon in taxons:
        taxon = taxon.strip()
        if not taxon:
            continue

        kingdom = get_kingdom(taxon)
        if kingdom:
            kingdom_taxons[kingdom].append(taxon)

    selected_taxons = []
    kingdom_summary = {}

    for kingdom, taxon_list in kingdom_taxons.items():
        sample_size = min(500, len(taxon_list))
        kingdom_summary[kingdom] = sample_size
        selected_taxons.extend(random.sample(taxon_list, sample_size))

    summary = [f"{kingdom}: {count}" for kingdom, count in kingdom_summary.items()]

    with open(output_file, 'w') as outfile:
        outfile.write("Selected taxon summary:\n")
        outfile.write('\n'.join(summary))
        outfile.write("\n\nSelected taxon list:\n")
        outfile.write('\n'.join(selected_taxons))


input_file = "149_prokaryotes.txt"
output_file = "149_prokaryotes_kingdoms.txt"
process_text_file(input_file, output_file)

#input_file = 'fix.txt'  # Replace with your input file name
#output_file = 'fixed_prokaryotes.txt'  # Replace with your output file name
#remove_spaces(input_file, output_file)


#fix_file()
#extract_taxons("prokaryotes_taxons2.nwk", "prokaryotes_taxons2.txt")