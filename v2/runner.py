from gen import run
from clean import gen_sequences,save_doc
from train import train,params
import numpy as np
def run_clean(names):
    files = names
    train_p = 100
    print(len(files))
    if type(files) != tuple and type(files) != list:
        files= [files]
    sequences = []
    print(files)
    for file in files:
        gen_sequences(file,sequences)
    sequences = np.array(sequences)
    #shuffle(sequences)
    lines = []
    for i in sequences:
        lines.append(' '.join(i))
    name = files[0]
    if len(files) > 1:
        name = "group"
    train_range = (len(sequences)*train_p) // 100
    out_filename = name+'_seq.txt'
    save_doc(lines[:train_range], out_filename)
    if train_p < 100:
        save_doc(lines[train_range:],name+'_val.txt')

overall_names = ["cat","cp","echo","findmnt","git","ls","mkdir"]
k=120
metric = 'euclidean'

for i in range(len(overall_names)):
    name = overall_names[i]
    names = [overall_names[j] for j in range(len(overall_names)) if j != i]
    print(name,names)
    run(names,name,k,metric=metric,save=True,viz=False)
    run_clean(names)
    run_clean(name)
    train(name,params,log_file=name)
  