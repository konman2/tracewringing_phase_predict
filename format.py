import sys
name = sys.argv[1]
f = open("in.trace",'r')
f2 = open(name+'.trace','w')
for line in f:
    line = line.strip()
    if line[0] == 'M':
        line = line.replace(',',' ')
        line = line.split(" ")
        #print(line)
        line[1] = "0x"+line[1]
        #print(line[1])
        l = " ".join(line)
        f2.write(l+'\n')
f.close()
f2.close()


