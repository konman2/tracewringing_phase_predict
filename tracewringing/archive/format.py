
f = open('./workload/gzip.trace','r')
f2 = open('gzip2.trace','w')
for line in f:
    f2.write(line.replace(',' ,' '))
f.close()
f2.close()

