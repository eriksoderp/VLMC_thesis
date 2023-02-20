import sys, getopt
import random
import os

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

    sequences = modify(input_file, number_of_letters, number_of_sequences)
    file_name = "modified_sequences/" + output_file

    
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w") as f:
        f.write(sequences)
    f.close()

def split(input_file):
    sdsd = []
    with open(input_file, 'r') as f:
        total = f.read()
        sdsd = total.split('\n>')
    f.close()
    print(sdsd[0])


# number_of_letters specify how many letters to replace
def modify(input_file, number_of_letters, number_of_sequences): 
    with open(input_file, 'r') as f:
        _ = next(f) # skips the description of the genome
        sequence = f.read() # parses sequence
        sequence = sequence.split('\n>')[0] # takes the first genome in a multifasta file
    f.close()

    sequence = sequence.strip()
    letters = ['A', 'C', 'G', 'T']*number_of_letters
    indices = [i for i, _ in enumerate(sequence)]
    sequences = ""

    for i in range(number_of_sequences):
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
        new_sequence = ">Number of dissimilarities is " + str(count) + "\n" + ''.join(lst) + '\n'

        sequences += new_sequence

    return sequences

if __name__ == "__main__":
    main(sys.argv[1:])
