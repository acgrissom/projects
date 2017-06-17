import sys, codecs
f = open("../pos.csv","r")
o = codecs.open("../../../../data/pos.test1.csv", "w",encoding="utf-8")
for line in f:
	if line.find("test1") > 0:
		o.write(unicode(line,"utf-8").encode("utf-8"))
o.close()
