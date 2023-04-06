import dendropy
import csv
import pandas as pd

def get_matrix_csv(newick_file_path, csv_file_path):
    
    tree = dendropy.Tree.get(path=newick_file_path, schema='newick')
    distance_matrix = tree.phylogenetic_distance_matrix()
    labels = [taxon.label for taxon in tree.taxon_namespace]

    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        
        header_row = [''] + labels
        csv_writer.writerow(header_row)
        
        for taxon1 in tree.taxon_namespace:
            row = [taxon1.label]
            for taxon2 in tree.taxon_namespace:
                distance = distance_matrix(taxon1, taxon2)/2
                row.append(distance)
            csv_writer.writerow(row)

    print(f"Distance matrix written to '{csv_file_path}'")


def get_vector(newick_file_path, output_csv_path):
    tree = dendropy.Tree.get(path=newick_file_path, schema='newick')
    distance_matrix = tree.phylogenetic_distance_matrix()

    vector_data = []

    # Create the three-column vector as [taxon1, taxon2, distance]
    for taxon1 in tree.taxon_namespace:
        for taxon2 in tree.taxon_namespace:
            distance = distance_matrix(taxon1, taxon2) / 2
            vector_data.append([taxon1.label, taxon2.label, distance])

    df_vector = pd.DataFrame(vector_data, columns=['taxon1', 'taxon2', 'distance'])

    # Save the DataFrame to a CSV file
    df_vector.to_csv(output_csv_path, index=False)

    return df_vector

    
    #return df_vector

### EXEMPEL
# get_matrix_csv('21_primates.nwk', 'distance_matrix.csv')
vector_df = get_vector('149_prokaryotes.nwk', '149_prokaryotes.csv')
# print(f"Vector as DataFrame: '{vector_df}")