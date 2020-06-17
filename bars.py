import matplotlib.pyplot as plt
import pickle
import matplotlib
import numpy as np



def make_bars(metrics,labels,names,params=None,title='Performance of Model on Program'):
    x = np.arange(len(names))  # the label locations
    width = 0.1  # the width of the bars
    fig, ax = plt.subplots()
    for i in range(len(labels)):
        # print(len(labels)//2-i,metrics[i])
        if params is None:
            ax.bar(x-(len(labels)//2-i)*width,metrics[i],width,label=labels[i])
        else:
            ax.bar(x-(len(labels)//2-i)*width,metrics[i],width,label=labels[i], **params[i])
    lim = ax.get_ylim()
    ax.set_ylim([0,lim[1]+0.3])
    ax.set_ylabel(title,fontsize=18)
    ax.set_title(title,fontsize=18)
    ax.set_xticks(x)
    #ax.set_yscale()
    ax.set_xticklabels(names,fontsize=18)
    ax.legend(fontsize=14)
    fig.tight_layout()

if __name__ == '__main__':
    # names,scores= list(pickle.load(open('model_performance.pkl','rb')))
    names,scores= list(pickle.load(open('mp2.pkl','rb')))
    # names = scores[0]
    # scores = scores[2:]
    # print(names,scores)

    metric_names=('precision at 5',
    'recall at 5', 
    'transition precision at 5', 
    'top5',
    'distance from perfect lstm', 
    'theoretical best(perfect lstm)',
    'distance from original',
    'mode distance from original',
    'white background')

    metric_names = ['precision lstm','precision mode', 'precision markov chain']
    # print(scores)
    # params=[{'hatch': '//', 'alpha':0.5}]
    # for i in metric_names[4:]:
    #     params.append({})
    # make_bars(scores[:4],metric_names[:4],names,title="Prediction Performance of Model on Program")
    # #plt.figure()
    # make_bars(scores[4:],metric_names[4:],names, params, title='Image distance of Model on Program')
    make_bars(scores,metric_names,names,title='LSTM comparison')
    plt.show()









