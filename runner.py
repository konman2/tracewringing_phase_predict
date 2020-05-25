from .gen import run
from .clean import gen_sequences,save_doc,clean
from .train import train,params
import numpy as np


overall_names = ["cat","cp","echo","findmnt","git","ls","mkdir"]
k=120
metric = 'euclidean'

for i in range(len(overall_names)):
    name = overall_names[i]
    names = [overall_names[j] for j in range(len(overall_names)) if j != i]
    print(name,names)
    run(names,name,k,metric=metric,save=True,viz=False)
    clean(names)
    clean(name)
    train(name,params,log_file=name)