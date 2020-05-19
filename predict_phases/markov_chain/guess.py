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

def calc_success(val_lines):
    name = ""
    prev = None
    count = 0
    total = 0
    transition_points = 0
    for i in val_lines:
        if prev != None:
            if guess(prev) == i:
                count+=1
            if prev != i:
                transition_points+=1 

            total+=1
        prev = i
    print(count/total)
    print("transition_points:",transition_points,"total: ",total)
    print(lib)

def predict(first_word, n_words):
    prev = first_word
    result  =[first_word]
    for i in range(n_words):
        a = guess(prev)
        result.append(a)
        prev = a
    return result


val_filename = "../phases/mkdir.phase"
val_doc = load_doc(val_filename)
val_lines = val_doc.split("\n")
# print(val_lines)
result = predict(val_lines[0],len(val_lines))
print(result)
print(len(result))
calc_success(val_lines)