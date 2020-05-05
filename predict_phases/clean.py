import string
import numpy as np
from numpy.random import shuffle
file = "gcc-1B"
# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.readlines()
	# close the file
	file.close()
	#print(text)
	return text
 
# # turn a doc into clean tokens
# def clean(doc):
# 	# train_range = (len(doc)*train_p) // 100
#	tokens = [i.split(' ')[1] for i in doc]
# 	# print(train_range,(len(tokens)-train_range)/2)
# 	return tokens
# 	#return tokens[:train_range],tokens[train_range:train_range+(len(tokens)-train_range)//2],tokens[train_range+(len(tokens)-train_range)//2:]
 
# # save tokens to file, one dialog per line
def save_doc(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()
 
# load document
in_filename = '../'+file+'.phases'
doc = load_doc(in_filename)

tokens = [i.split(' ')[0][0] for i in doc]

print(tokens[:200])
print('Total Tokens: %d' % len(tokens))
print('Unique Tokens: %d' % len(set(tokens)))

# # organize into sequences of tokens
length = 15
length+=1
sequences = list()

for i in range(length, len(tokens)):
	seq = tokens[i-length:i]
	# if tokens[i][2] == '5':
	# 	seq[-1] = 0
	# elif tokens[i][2] == '7':
	# 	seq[-1] = 1
	# else:
	# 	print("oof")
	#line = ' '.join(seq)
	sequences.append(seq)
print('Total Sequences: %d' % len(sequences))

# save sequences to file

sequences = np.array(sequences)
shuffle(np.array(sequences))
lines = []
for i in sequences:
	lines.append(' '.join(i))


#print(lines)
train_p = 80
train_range = (len(sequences)*train_p) // 100
out_filename = file+'_seq.txt'
save_doc(lines[:train_range], out_filename)
save_doc(lines[train_range:train_range+(len(sequences)-train_range)//2],file+'_val.txt')
save_doc(lines[train_range+(len(sequences)-train_range)//2:],file+"_test.txt")