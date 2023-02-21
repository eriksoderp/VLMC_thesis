import sys, getopt
import random
import os
import numpy as np
import pandas as pd

def main(argv):
    input_file = ''
    output_file = ''
    number_of_sequences = 1
    number_of_letters = 10
    arg_help = 'modify.py -i <inputfile> -s <number_of_sequences> -l <number_of_letters> -o <outputfile>'
    try:
        opts, args = getopt.getopt(argv, "hi:s:l:o:", ["ifile=","number_of_letters=","ofile="])
    except:
        print(arg_help)
        sys.exit()
    
    for opt, arg in opts:
        if opt == '-h':
            print(arg_help)
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-s", "--number_of_sequences"):
            number_of_sequences = int(arg)
        elif opt in("-l", "--number_of_letters"):
            number_of_letters = int(arg)
        elif opt in ("-o", "--ofile"):
            output_file = arg
        

    print ('Input file is', input_file)
    print ('Number of sequences is', number_of_sequences)
    print ('Number of letters is', number_of_letters)
    print ('Output file is', output_file)

    first_part = output_file.split('.')[0]
    sequences, e_distances = modify(input_file, number_of_letters, number_of_sequences)
    e_distances = np.asarray(e_distances)

    distances_file_name = "modified_sequences/" + "distances_" + first_part + ".csv"
    os.makedirs(os.path.dirname(distances_file_name), exist_ok=True)
    pd.DataFrame(e_distances).to_csv(distances_file_name)
    
    file_name = "modified_sequences/" + output_file
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w") as f:
        f.write(sequences)
    f.close()


# number_of_letters specify how many letters to replace
def modify(input_file, number_of_letters, number_of_sequences): 
    with open(input_file, 'r') as f:
        description = next(f) # skips the description of the genome
        sequence = f.read() # parses sequence
        sequence = sequence.split('\n>')[0] # takes the first genome in a multifasta file
    f.close()

    sequence = sequence.strip()
    letters = ['A', 'C', 'G', 'T']*number_of_letters
    indices = [i for i, _ in enumerate(sequence)]
    sequences = ">Original \n" + sequence + "\n"
    seqs = [([], sequence)]
    result = [[0]*(number_of_sequences+1) for _ in range(number_of_sequences+1)]

    for _ in range(number_of_sequences):
        count = 0
        new_letters = iter(random.sample(letters, number_of_letters))
        sam = random.sample(indices, number_of_letters)
        lst = list(sequence)
        for i in sam:
            c = next(new_letters)
            if lst[i] == c:
                count += 1
            else:
                lst[i] = c

        count = number_of_letters - count
        seq = ''.join(lst)
        seqs.append((sam, seq))
        new_sequence = ">Number of dissimilarities from original is " + str(count) + "\n" + seq + "\n"

        sequences += new_sequence

    passed = set()
    for i, (ids1, s1) in enumerate(seqs):
        for j, (ids2, s2) in enumerate(seqs):
            if (j,i) in passed:
                continue
            ids = set(ids1)
            if i != j:
                ids.update(ids2)
                d = get_evolutionary_distance(s1, s2, ids)
                result[i][j] = d
                result[j][i] = d

            passed.add((i,j))
    return sequences, result

# get the evolutionary distance between two equally long sequences
def get_evolutionary_distance(s1, s2, ids):
    count = 0
    for i in ids:
        if s1[i] != s2[i]:
            count += 1
    return count

if __name__ == "__main__":
    main(sys.argv[1:])
