import sys
import os
import subprocess

# output file is out.tok
# run this command before (from the sockeye tutorial)
# sed -re 's/(@@ |@@$)//g' <out.bpe >out.tok

from nltk.translate.ribes_score import sentence_ribes

from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

#def translate():
 #   os.system(f"subword-nmt apply-bpe -c codes <test.{source_lang} >test.{source_lang}.bpe")
  #  subprocess.run(["sockeye-translate", "--input", f"test.{source_lang}.bpe","--output", "out.bpe","--model", "model", "--dtype", "float16", "--beam-size", "5","--batch-size", "64"])
   # os.system("sed -re 's/(@@ |@@$)//g' <out.bpe >out.tok")


def get_ribes_score(test_file : str, out_file : str):
    reference_lines = []

    with open(test_file) as f:
        for line in f:
            line = line.lower()
            new_line = tokenizer.tokenize(line)
            reference_lines.append(new_line)


    candidate_lines = []

    with open(out_file) as f:
        for line in f:
            line = line.lower()
            new_line = tokenizer.tokenize(line)
            candidate_lines.append(new_line)


    scores = []

    for i in range(len(reference_lines)):
        ref = [reference_lines[i]]
        can = candidate_lines[i]

        try:
            score = sentence_ribes(ref, can)
        except ZeroDivisionError:
            score = 0
            print(ref)
            print(can)
            print(score)
        
        scores.append(score)
        

    print(sum(scores)/ len(scores))
    avg = sum(scores)/ len(scores)
    return avg

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please enter the target test file")
        
    else:
        target_test = sys.argv[1]
        score = get_ribes_score(target_test, "out.tok")
        print(score)

