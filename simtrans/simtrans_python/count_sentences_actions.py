filename = "results/denews_actions_rewards-josh-optimal-bw4.csv"
f = open(filename)
ids = dict()
for line in f:
    line_toks = line.split(",")
    if not line_toks[0] in ids:
        ids[line_toks[0]] = line_toks[0]
    
print str(len(ids))
