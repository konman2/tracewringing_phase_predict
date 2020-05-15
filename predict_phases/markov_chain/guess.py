import numpy as np
import pickle
import sys
lib = pickle.load(open('lib2.pkl','rb'))

def guess(w):
    if w not in lib:
        return "No words follow"
    words,p = lib[w]
    return np.random.choice(words, p=p)

def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

name = ""
val_filename = "../mkdir.phases"
val_doc = load_doc(val_filename)
val_lines = val_doc.split("\n")
print(val_lines)
prev = None
count = 0
total = 0
for i in val_lines:
    if prev != None:
        if guess(prev) == i:
            count+=1
        total+=1
    prev = i
print(count/total)

# def split_X_y(lines):
#     sequences = []
#     for line in lines:
#         l = []
#         for word in line.split(' '):
#             l.append(word)
#         sequences.append(l)
#     sequences = np.array(sequences)
#     return sequences[:,-2],sequences[:,-1]

# X,y = split_X_y(val_lines)
# print(X)
# count = 0
# for i in range(len(y)):
#     #print(X[i])
#     if guess(X[i]) == y[i]:
#         count+=1
# print(count/len(y))