from gen import gen
from clean import gen_sequences,save_doc,clean
from train import train,params
import numpy as np
import torch
import pickle
from markov_chain.markov import markov
from markov_chain.compute_prob import compute
from markov_chain.guess import guess1
import names
overall_names = ["cat","cp","echo","findmnt","git","ls","mkdir"]

k=120
metric = 'euclidean'
average_cluster = []
test_name = []
transition_precision = []
precision = []
recall = []
base_model_precision = []
score = []
perfect_score = []
original_score = []
mode_score = []
background_score = []
metric_names=['precision at 5','recall at 5', 'transition precision at 5', 'top5','distance from perfect lstm', 'distance from original','theoretical best(perfect lstm)','mode distance from original','white background']
#metric_names = ['precision lstm','precision mode', 'precision markov chain']
metrics =[[] for i in range(len(metric_names))]

for i in range(len(overall_names)):
    name = overall_names[i]
    names = [overall_names[j] for j in range(len(overall_names)) if j != i]
    print(name,names)
    # gen(names,name,k,metric=metric,save=False,viz=True,verbose=True)
    test_name.append(name)
    gen(names,name,k,metric=metric,save=False,viz=True,verbose=True,log_file=name)
    clean(names,rle=False)
    clean(name,rle=False)
    # markov(names)
    # compute()
    # result = guess1(name)

    # results = train(name,params,log_file=True)
    # p,r,tp,b,bs,ps,bos,ms,aws = results
    # results = (results[0],results[1],result)
    # print(results)
    # for c,i in enumerate(results):
    #     metrics[c].append(i)
    # precision.append(p)
    # recall.append(r)
    # transition_precision.append(tp)
    # base_model_precision.append(b)
    # score.append(bs)
    # perfect_score.append(ps)
    # original_score.append(bos)
    # mode_score.append(ms)
    # background_score.append(aws)
    print(metrics)

# print(test_name)
# print(average_cluster)
# print(precision)
# print(recall)
# print(transition_precision)
# print(base_model_precision)
# print(precision)
# print(recall)
# print(transition_precision)
# print(base_model_precision)
# print(score)
# print(perfect_score)
# print(original_score)
# print(mode_score)
# print(background_score)
print(metrics)

#pickle.dump((test_name,average_cluster,precision,recall,transition_precision,base_model_precision,score,perfect_score,original_score,mode_score,background_score),open("model_performance.pkl",'wb'))
# pickle.dump((overall_names,metrics),open("model_performance.pkl",'wb'))
#pickle.dump((overall_names,metrics),open("mp3.pkl",'wb'))
#pickle.dump((overall_names,metrics),open("mp2.pkl",'wb'))
    # print(torch.cuda.memory_allocated())
    # torch.cuda.empty_cache()
    # print(torch.cuda.memory_allocated())