from gen import *
sse={}
for k in range(1,20):
    labels,centroids,kmeans,heatmap=gen_cluster("mkdir",k,False)
    print(k,kmeans.inertia_)
    sse[k] = kmeans.inertia_
plt.plot(list(sse.keys()),list(sse.values()))
plt.xlabel("clusters")
plt.ylabel("SSE")
plt.show()