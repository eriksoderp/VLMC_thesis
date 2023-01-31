import sys, getopt
import random
import os

def main(argv):
    input_file = ''
    output_file = ''
    number_of_letters = 10
    arg_help = 'test.py -i <inputfile> -n <number_of_letters> -o <outputfile>'
    try:
        opts, args = getopt.getopt(argv, "hi:n:o:", ["ifile=","number_of_letters=","ofile="])
    except:
        print(arg_help)
        sys.exit()
    
    for opt, arg in opts:
        if opt == '-h':
            print(arg_help)
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in("-n", "--number_of_letters"):
            number_of_letters = int(arg)
        elif opt in ("-o", "--ofile"):
            output_file = arg
        

    print ('Input file is', input_file)
    print ('Number of letters is', number_of_letters)
    print ('Output file is', output_file)

    new_sequence = modify(input_file, number_of_letters)
    file_name = "modified_sequences/" + output_file
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w") as f:
        f.write(new_sequence)
    f.close()

# number_of_letters specify how many letters to replace
def modify(input_file, number_of_letters): 
    with open(input_file, 'r') as f:
        description = next(f) # saves the description line
        sequence=f.read() # parses sequence
    f.close()

    sequence = sequence.strip()
    letters = ['A', 'C', 'G', 'T']*number_of_letters
    indices = [i for i, _ in enumerate(sequence)]
    new_letters = iter(random.sample(letters, number_of_letters))
    sam = random.sample(indices, number_of_letters)
    lst = list(sequence)
    for i in sam:
        lst[i] = next(new_letters)

    new_sequence = description + ''.join(lst)

    return new_sequence

if __name__ == "__main__":
    main(sys.argv[1:])
