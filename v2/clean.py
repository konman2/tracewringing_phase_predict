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
 
# # turn a doc into clean tokens
# def clean(doc):
# 	# train_range = (len(doc)*train_p) // 100
#	tokens = [i.split(' ')[1] for i in doc]
# 	# print(train_range,(len(tokens)-train_range)/2)
# 	return tokens
# 	#return tokens[:train_range],tokens[train_range:train_range+(len(tokens)-train_range)//2],tokens[train_range+(len(tokens)-train_range)//2:]
 
def save_doc(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

def gen_sequences(file,sequences):
	in_filename = "phases/"+file+'.phase'
	doc = load_doc(in_filename)
	print(file)
	tokens = [i.strip() for i in doc]
	
	print(tokens[:200])
	print('Total Tokens: %d' % len(tokens))
	print('Unique Tokens: %d' % len(set(tokens)))

	# # organize into sequences of tokens
	length = seq_len
	length+=1
	#sequences = []
	curr_seq = []
	for i in range(length, len(tokens)):
		seq = tokens[i-length:i]
		curr_seq.append(seq)
		sequences.append(seq)
	print('Total Sequences: %d' % len(curr_seq))

	return sequences


if __name__ == "__main__":
	files = gen.names
	train_p = 100
	if len(sys.argv) > 1:
		files = [sys.argv[1]]
		train_p =100
	sequences = []
	print(files)
	for file in files:
		gen_sequences(file,sequences)

	sequences = np.array(sequences)

	#shuffle(sequences)
	lines = []

	for i in sequences:
		lines.append(' '.join(i))


	#print(lines)
	name = files[0]
	if len(files) > 1:
		name = "group"
	train_range = (len(sequences)*train_p) // 100
	out_filename = name+'_seq.txt'
	save_doc(lines[:train_range], out_filename)
	if train_p < 100:
		save_doc(lines[train_range:],name+'_val.txt')
	#save_doc(lines[train_range:train_range+(len(sequences)-train_range)//2],file+'_val.txt')
	#save_doc(lines[train_range+(len(sequences)-train_range)//2:],file+"_test.txt")