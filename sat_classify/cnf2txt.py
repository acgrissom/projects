import sys
import os

def convert_to_text(filename):
    f = open(filename)
    i = 0
    cleaned = ''
    for line in f:
        if line.startswith('p') or line.startswith('c') or line.startswith('%') or line.strip() == '0':
            i += 1
            continue
        i += 1
        line = line.strip()
        cleaned += ' '.join([x for x in line.split()[0:3]]) + '\n'
    return cleaned

def convert_to_sparse_binary(filename, num_variables=50):
    binary_encoding = ''
    cleaned = convert_to_text(filename)
    cleaned = cleaned.split('\n')
    for line in cleaned:
        cols = line.split()
        new_cols = [str(0)] * (num_variables + 1)
        for i in cols:
            k = int(i)
            new_cols[abs(k)] = str(1);
            if k < 0:
                new_cols[abs(k)] = str(-1);
        binary_encoding += ' '.join(new_cols) + '\n'
    return binary_encoding
    
def convert_dir(in_dir, out_dir, sat_or_unsat, encoding='sparse_binary'):
    converted = None
    for filename in os.listdir(in_dir):
        if filename.endswith(".cnf"):
            if encoding == 'dense_matrix':
                converted = convert_to_text(in_dir + '/' + filename)
            elif encoding == 'sparse_binary':
                converted = convert_to_sparse_binary(in_dir + '/' + filename)
            print(filename)
            try:  
                os.mkdir(out_dir)
            except OSError:  
                pass
            #    print ("Creation of the directory %s failed" % out_dir)
            #else:  
            #    print ("Successfully created the directory %s " % out_dir)
            f = open(out_dir + '/' + filename + '.' + sat_or_unsat, 'w')
            f.write(converted)
            f.close()


#convert_dir('./CBS_k3_n100_m403_b10/','sat_files', 'sat')
#convert_dir('./dubois/','sat_files', 'unsat')
convert_dir('./data/uf50/unsat','sat_files', 'unsat')
convert_dir('./data/uf50/sat','sat_files', 'sat')
