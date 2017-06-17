#!/opt/local/stow/python-2.7.2/bin/python -S

import sys
import csv
import codecs
import locale
from io import BytesIO
from collections import defaultdict
#sys.stdout = codecs.getwriter(locale.getpreferredencoding())(sys.stdout) 
#sys.stdin = codecs.getwriter(locale.getpreferredencoding())(sys.stdin)
#reload(sys) 
outfile = open("../pos.vf.csv", "w")
poswriter = csv.writer(outfile)
last_toks = defaultdict(int)
with open("../pos.ascii.csv") as csvfile:
    csv_reader = csv.reader(csvfile)
    row_num = 0
    for row in csv_reader:
        row_num += 1
        if row_num == 1:
            poswriter.writerow(row)
            continue
        words = row[3].split()
        #print words[len(words) - 2
        parsed_tok = words[len(words)-2]
        if parsed_tok.split("_") < 2:
            continue
        pos = parsed_tok.split("_")[1]
        last_toks[pos] += 1
        if pos[0] == "V":
            poswriter.writerow(row)

print last_toks
#poswriter.close()
outfile.close()
        #print row[3]
        #print words[len(words)]


    
    
