from mecab import MeCab
import sys

mecab = MeCab()

r_file = open(sys.argv[1], 'r')
w_file = open(sys.argv[2], 'w')

lines = r_file.readlines()

for line in lines:
  morph = mecab.morphs(line)
  n_line = ' '.join(morph)
  w_file.write(n_line + "\n")

r_file.close()
w_file.close() 
