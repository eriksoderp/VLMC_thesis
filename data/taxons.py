import re

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

extract_taxons("taxa-eukaryotes.nwk", "eukaryotes_taxons.txt")