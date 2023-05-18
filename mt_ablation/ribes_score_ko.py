import sys
import os
import subprocess

from nltk.translate.ribes_score import sentence_ribes
import mecab_ko as MeCab
import mecab_ko_dic
tagger = MeCab.Tagger(mecab_ko_dic.MECAB_ARGS + " -Owakati")

#def translate():
 #   os.system(f"subword-nmt apply-bpe -c codes <test.{source_lang} >test.{source_lang}.bpe")
  #  subprocess.run(["sockeye-translate", "--input", f"test.{source_lang}.bpe","--output", "out.bpe","--model", "model", "--dtype", "float16", "--beam-size", "5","--batch-size", "64"])
   # os.system("sed -re 's/(@@ |@@$)//g' <out.bpe >out.tok")


def get_ribes_score(test_file : str, out_file : str):
    reference_lines = []

    with open(test_file) as f:
        for line in f:
            line = line.strip()
            new_line = tagger.parse(line).split()
            reference_lines.append(new_line)

    candidate_lines = []

    with open(out_file) as f:
        for line in f:
            line = line.strip()
            new_line = tagger.parse(line).split()
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
        
    avg = sum(scores)/ len(scores)
    return avg

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please enter the target test file")
    else:
        target_test = sys.argv[1]
        score = get_ribes_score(target_test, "out.tok")
        print(score)




