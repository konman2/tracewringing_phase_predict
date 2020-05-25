import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import sys
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
name = "gzip_mem_1_10000_2048_1.csv"
name2 = "opt.csv"
df = pd.read_csv("/home/mkondapaneni/Research/tracewringing/sweep/"+name)

# bound = [
#     (1,60),     # thres, 
#     (1,60),     # gap, 
#     (1,60),     # length, 
#     (1,60),     # blocksize, 
#     (25,99),    # fp (filter percent)
#     (2,5)       # clusters
# ]

X = np.array([i.split('_') for i in df['param_id']],dtype=int)
y = np.array(df['rel_err'])
y2 = np.array(df['mr'])

X_tsne = TSNE(n_components=2,verbose=1,n_iter=1000,random_state=2).fit_transform(X)
model = PCA(n_components=2).fit(X)
#model = PCA().fit(X)
X_pca = model.transform(X)
n_pcs = model.components_.shape[0]
most_important = [np.abs(model.components_[i]).argmax() for i in range(n_pcs)]
initial_feature_names = ['thres','gap','length','blocksize','filter percent','clusters']
most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]
dic ={'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}
df = pd.DataFrame(dic.items())
print(df)

def graph(X_embedd,y,param,axes):
    fig,ax = plt.subplots()
    
    print("max " +param+ "=",np.max(y))
    plot = ax.scatter(X_embedd.T[0:1].T,X_embedd.T[1:].T,c=y/np.max(y),cmap=plt.cm.tab20b,vmin=0,vmax=1)
    cbar = fig.colorbar(plot)
    cbar.set_label('max_scaled '+ param)
    ax.set_xlabel(axes[0])
    ax.set_ylabel(axes[1])

print(len(y))

# ax1= ('thres','gap')
# ax2 = ('PC1','PC2')
# ax3 = ('t-sne1','t-sne2')
# X_highlight = np.array([X.T[0],X.T[1]]).T
# graph(X_highlight,y,"relative error",ax1)
# graph(X_highlight,y2,"bit leakage",ax1)
# graph(X_pca,y2,"bit leakage",ax2)
# graph(X_pca,y,"relative error",ax2)
# graph(X_tsne,y2,"bit leakage",ax3)
# graph(X_tsne,y,"relative error",ax3)
fig = plt.figure()
ax = Axes3D(fig)
plot = ax.scatter(X.T[0].T,X.T[1].T,X.T[2].T,c=y/np.max(y),cmap=plt.cm.tab20b,vmin=0,vmax=1)
cbar = fig.colorbar(plot)
cbar.set_label('max scaled error')
ax.set_xlabel('threshold')
ax.set_ylabel('gap')
ax.set_zlabel('length')
fig = plt.figure()
ax = Axes3D(fig)
plot = ax.scatter(X.T[0].T,X.T[1].T,X.T[2].T,c=y2/np.max(y2),cmap=plt.cm.tab20b,vmin=0,vmax=1)
cbar = fig.colorbar(plot)
cbar.set_label('max scaled bit leakage')
ax.set_xlabel('threshold')
ax.set_ylabel('gap')
ax.set_zlabel('length')

plt.show()




# fig,ax = plt.subplots()
# x_plot = np.arange(y.shape[0])
# ax.bar(x_plot,y/np.max(y),color='b',label='Relative Error',alpha=0.5)
# ax.bar(x_plot,y2/np.max(y2),color = 'y',label = 'Bits Leaked',alpha=0.5)
# #ax.scatter(x_plot,abs(y/np.max(y)-y2/np.max(y2)))

# plt.title('Max Scaled Relative Error vs Bits Leaked')
# ax.legend()

# #plt.show()




