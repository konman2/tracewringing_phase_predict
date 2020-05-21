from gen import run

overall_names = ["cat","cp","echo","findmnt","git","ls","mkdir"]
k=120
metric = 'euclidean'

for i in range(len(overall_names)):
    name = overall_names[i]
    names = [overall_names[j] for j in range(len(overall_names)) if j != i]
    print(name,names)
    run(names,name,k,metric=metric,save=True)
