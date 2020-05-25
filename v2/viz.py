import pickle
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

addresses = pickle.load(open('words.pkl','rb'))
val = pickle.load(open('val.pkl','rb'))
print(addresses[0:10])
addresses = addresses[1:]
# addresses = np.array(addresses).reshape(-1,1)
# print(addresses[0:10])
# kmeans = KMeans(n_clusters=2,random_state=2).fit(addresses)
#plt.scatter(kmeans.labels_,addresses)
# plt.scatter([i for i in range(len(addresses))],addresses)
plt.plot(val)
plt.show()