
f = open('./gzip.trace','r')
f2 = open('gzip2.trace','w')
f3 = open('gzip3.trace','w')
for line in f:
    f2.write(line[1:])
    f3.write(line[2:len(line)-3]+' ')
f.close()
f2.close()
f3.close()

