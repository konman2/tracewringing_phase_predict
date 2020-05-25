import string
import numpy as np
from numpy.random import shuffle
import sys
import gen

seq_len = 120
# load doc into memory
def load_doc(filename):
    file = open(filename, 'r')
    text = file.readlines()
    file.close()
    return text
 
def encodeRLE(labels, collapse_factor):
    """Run-length encoding for cluster labels. """
    tokens = []
    lengths = []
    count = 0
    i = 0
    character = labels[i]
    for i in range(0,len(labels)):
        if(labels[i] == character):
            count += collapse_factor
        else:
            tokens.append(character)
            lengths.append(str(count))
            character = labels[i]
            count = collapse_factor
        if(i==(len(labels)-1)):
            tokens.append(character)
            lengths.append(str(count))
    return tokens,lengths
 
def save_doc(lines, filename):
    data = '\n'.join(lines)
    file = open("sequences/"+filename, 'w')
    file.write(data)
    file.close()

def gen_sequences(file,sequences,sequence_lens,test):
    in_filename = "phases/"+file+'.phase'
    doc = load_doc(in_filename)
    print(file)
    print(doc)
    tokens = [i.strip() for i in doc]
    enc,lens = encodeRLE(tokens,1)
    # print(tokens[:200])
    # print(enc[:200])
    # print(lens[:200])
    print('Total Tokens: %d' % len(tokens))
    print('Unique Tokens: %d' % len(set(tokens)))

    # # organize into sequences of tokens
    length = seq_len
    length+=1
    #sequences = []
    seq_count = 0
    for i in range(length, len(tokens)):
        seq = tokens[i-length:i]
        test.append(seq)
    for i in range(length, len(enc)):
        seq = enc[i-length:i]
        seq2 = lens[i-length:i]
        seq_count += 1
        sequences.append(seq)
        sequence_lens.append(seq2) 
    print('Total Sequences: %d' % seq_count)

    return sequences,sequence_lens

def clean(names):
    files = names
    train_p = 100
    print(len(files))
    if type(files) != tuple and type(files) != list:
        files= [files]
    sequences = []
    sequence_lens= []
    test = []
    print(files)
    for file in files:
        gen_sequences(file,sequences,sequence_lens,test)
    #print(test)
    sequences = np.array(sequences)
    test = np.array(test)
    #shuffle(sequences)
    test_lines = [' '.join(i) for i in test]
    lines = [' '.join(i) for i in sequences]
    lines2 = [' '.join(i) for i in sequence_lens]
    #print(lines)
    #print(lines2)
    # for i in sequences:
    #     lines.append(' '.join(i))
    
    name = files[0]
    if len(files) > 1:
        name = "group"
    test_train_range = (len(test)*train_p) // 100
    train_range = (len(sequences)*train_p) // 100
    out_filename = name+'_seq.txt'
    out_filename2 = name+'lens_seq.txt'
    save_doc(test_lines[:test_train_range],out_filename)
    # save_doc(lines[:train_range], out_filename)
    # save_doc(lines2[:train_range],out_filename2)
    # save_doc(sequence_lens,name+"_len_seq.txt")
    if train_p < 100:
        save_doc(lines[train_range:],name+'_val.txt')
        save_doc(lines2[:train_range],name+'lens_val.txt')
        

if __name__ == "__main__":
    files = gen.names
    if len(sys.argv) > 1:
        files = [sys.argv[1]]
    clean(files)