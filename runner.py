from gen import gen
from clean import gen_sequences,save_doc,clean
from train import train,params
import numpy as np
import torch


overall_names = ["cat","cp","echo","findmnt","git","ls","mkdir"]
k=120
metric = 'euclidean'

for i in range(len(overall_names)):
    name = overall_names[i]
    names = [overall_names[j] for j in range(len(overall_names)) if j != i]
    print(name,names)
    gen(names,name,k,metric=metric,save=True,viz=False,verbose=False,log_file=name)
    clean(names)
    clean(name)
    train(name,params,log_file=name)
    print(torch.cuda.memory_allocated())
    torch.cuda.empty_cache()
    print(torch.cuda.memory_allocated())