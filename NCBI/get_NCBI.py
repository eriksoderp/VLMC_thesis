import os
import gzip
from Bio import Entrez

def decompress_gz_file(gz_file_path, output_file_path):
    with gzip.open(gz_file_path, 'rb') as gz_file, open(output_file_path, 'wb') as output_file:
        output_file.write(gz_file.read())
    os.remove(gz_file_path)

def download_complete_genomes(genome_list, output_folder, email):
    Entrez.email = email

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for genome_name in genome_list:
        search_term = f"{genome_name}[Organism] AND (complete genome[Title] OR chromosome[Title])"
        handle = Entrez.esearch(db="assembly", term=search_term)
        record = Entrez.read(handle)
        handle.close()

        if record["Count"] == "0":
            print(f"Complete genome for {genome_name} not found.")
            continue

        assembly_id = record["IdList"][0]

        handle = Entrez.esummary(db="assembly", id=assembly_id)
        summary = Entrez.read(handle)
        handle.close()

        ftp_link = summary["DocumentSummarySet"]["DocumentSummary"][0]["FtpPath_GenBank"]

        if not ftp_link:
            print(f"GenBank FTP link not found for {genome_name}.")
            continue

        file_name = os.path.basename(ftp_link)
        ftp_link = ftp_link.replace("ftp://", "https://")
        gz_file_url = f"{ftp_link}/{file_name}_genomic.fna.gz"
        gz_output_file_path = os.path.join(output_folder, f"{file_name}_genomic.fna.gz")
        fna_output_file_path = os.path.join(output_folder, f"{file_name}_genomic.fna")

        print(f"Downloading complete genome for {genome_name}...")
        os.system(f"curl -L -o {gz_output_file_path} {gz_file_url}")

        #print(f"Downloaded complete genome for {genome_name} to {gz_output_file_path}")

        # Decompress the .gz file
        #print(f"Decompressing {gz_output_file_path}...")
        decompress_gz_file(gz_output_file_path, fna_output_file_path)
        #print(f"Decompressed {gz_output_file_path} to {fna_output_file_path}")

        # Print the size of the downloaded .fna file
        file_size = os.path.getsize(fna_output_file_path)
        print(f"Complete, size of {fna_output_file_path}: {file_size} bytes", "\n")

# Example usage



if __name__ == "__main__":
    
    genome_list = []

    # If you want to test the module
    #genome_list = ["Daubentonia madagascariensis"]
    
    with open("list.txt", "r") as file:
        lines = file.readlines()
        array = [line.strip() for line in lines]

    output_folder = "complete_genomes"
    email = "your_email@example.com"
    
    
    download_complete_genomes(genome_list, output_folder, email)