#Generates random matirces
import random
import sys
import numpy as np
from string import ascii_lowercase
import codecs
sequences = dict()

LETTERS = {str(index - 1) : letter for index, letter in enumerate(ascii_lowercase, start=1)} 
ANNOTATIONS = ['[', ']', '[[', ']]',' ']
#print(LETTERS)
def num2alpha(text):
    text = text.replace('[','').replace(']','')
    newtext = ''
    for c in text.split():
        if c in ANNOTATIONS:
            newtext += c
        elif c == '\n':
            continue
        else:
            newtext += LETTERS[c] + ' '
    return newtext

def generate_sequence(length,max_num, f_source, f_target):
    seq = ''   

    matrix = np.random.randint(max_num,size=(length,length))
    transpose = matrix.transpose()
    num = str(random.randint(0,max_num))
    matrix = matrix.flatten()
    transpose = transpose.flatten()
    str_matrix = ''
    for i in matrix:
        str_matrix += str(i) + ' '
    str_matrix = num2alpha(str_matrix)

    sequences[str_matrix] = str_matrix
    if seq in sequences:
        return
    str_transpose = ''
    for i in transpose:
        str_transpose += str(i) + ' '
    str_transpose = num2alpha(str_transpose)
    str_transpose = np.array2string(transpose)
    str_transpose = num2alpha(str_transpose.strip())
    f_source.write(str_matrix + '\n')
    f_target.write(str_transpose + '\n')

    #sys.stdout.write(str_matrix + '\t' + str_transpose)
    #sys.stdout.write('\n')


def generate_sequence_brackets(length,max_num, f_source, f_target):
    seq = ''  
    matrix = np.random.randint(max_num,size=(length,length))
    transpose = matrix.transpose()
    num = str(random.randint(0,max_num))
    #matrix = matrix.flatten()
    transpose = transpose.flatten()
    str_matrix = np.array2string(matrix)
    str_matrix = num2alpha(str_matrix.strip())
    sequences[str_matrix] = str_matrix
    if seq in sequences:
        return
    str_transpose = np.array2string(transpose)
    str_transpose = num2alpha(str_transpose.strip())
    f_source.write(str_matrix + '\n')
    f_target.write(str_transpose + '\n')

    #sys.stdout.write(str_matrix + '\t' + str_transpose)
    #sys.stdout.write('\n')


length = 9 #num of elements in rows and colums (all are square matrices)
max_num = 25 #number of distinct symbols
filename_base = 'transpose'
percent_dev = .1
percent_test = .1
percent_train = 1.0 - (percent_dev + percent_test)
total_num = 10000
test_source = codecs.open(filename_base + '.' + 'test' + '.source', encoding="utf8", mode='w')
test_target = codecs.open(filename_base + '.' + 'test' + '.target', encoding="utf8", mode='w')

dev_source = codecs.open(filename_base + '.' + 'dev' + '.source', encoding="utf8", mode='w')
dev_target = codecs.open(filename_base + '.' + 'dev' + '.target', encoding="utf8", mode='w')

train_source = codecs.open(filename_base + '.' + 'train' + '.source', encoding="utf8", mode='w')
train_target = codecs.open(filename_base + '.' + 'train' + '.target', encoding="utf8", mode='w')


for i in range(round(percent_train * total_num)):
    #generate_sequence_brackets(length, max_num)
    generate_sequence(length, max_num, train_source, train_target)
for i in range(round(percent_dev * total_num)):
    generate_sequence(length, max_num, dev_source, dev_target)
for i in range(round(percent_test * total_num)):
    generate_sequence(length, max_num, test_source, test_target)
    
train_source.close()
train_target.close()
dev_source.close()
dev_target.close()
test_source.close()
test_target.close()
