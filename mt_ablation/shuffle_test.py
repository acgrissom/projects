import random
import random
import sys
from pathlib import Path
import subprocess
import os

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

bleu_scores = []
ribes_scores = []

shuffle_values = [0, 0.2, 0.4, 0.6, 0.8, 1]

# shuffles a sentence to a specific percentage
def shuffle_sentence(s, pct_to_shuffle):
    original_tokens = s.split()

    indices_to_shuffle = random.sample(
        range(0, len(original_tokens)), round((len(original_tokens))*pct_to_shuffle))

    words_to_shuffle = [original_tokens[i] for i in range(
        len(original_tokens)) if i in indices_to_shuffle]

    random.shuffle(words_to_shuffle)

    final_sentence = ["" for i in range(len(original_tokens))]

    j = 0
    i = 0
    while i < len(original_tokens):
        if i not in indices_to_shuffle:
            final_sentence[i] = original_tokens[i]
        else:
            final_sentence[i] = words_to_shuffle[j]
            j += 1
        i += 1

    return final_sentence

# write shuffled sentences to a file
def output_shuffled_file(pct_to_shuffle, source_file, shuffled_output_file_name):
    all_lines = []

    print(f"Input file: {source_file}")
    print(f"Output file: {shuffled_output_file_name}")

    print(f"Reading input from {source_file}")

    # get the source lines
    with open(source_file) as file:
        for line in file:
            line_as_list = shuffle_sentence(line, pct_to_shuffle)
            line = " ".join(line_as_list)
            all_lines.append(line + "\n")

    print(
        f"Finished shuffling with percentage={pct_to_shuffle}, now writing file to {shuffled_output_file_name}")

    f = open(shuffled_output_file_name, "w")
    for line in all_lines:
        f.write(line)
    f.close()

# convert the shuffle file to bpe
def bpe_convert(shuffled_output_file_name, source_lang, val):
    bpe = f"bpe.shuffled_{val}_ko-en-corpus.test.{source_lang}"
    os.system(f"subword-nmt apply-bpe -c bpe_ko.codes --vocabulary  bpe.vocab.{source_lang} <  {shuffled_output_file_name} >  {bpe}")
    return bpe

# translate with model
def translate(bpe, source_lang, target_lang, model):

    subprocess.run(["python", "-m", "sockeye.translate", "--input", bpe,"--output", "out.bpe","--model", 
    model, "--dtype", "float16", "--beam-size", "5","--batch-size", "64"])

    os.system("sed -re 's/(@@ |@@$)//g' <out.bpe >out.tok")

    if target_lang == "ko":
        
        bleu = subprocess.run(["python", "ribes_score_ko.py", f"/share/kuran/data/joseph/korean_corpus/ko-en-corpus.test.{target_lang}"], 
                       capture_output=True, text=True)

        ribes = subprocess.run(["sacrebleu", f"/share/kuran/data/joseph/korean_corpus/ko-en-corpus.test.{target_lang}", 
                       "-tok", "ko-mecab", "-i", "out.tok", "-b"], capture_output=True, text=True)

    else:
        bleu = subprocess.run(["python", "ribes_score.py", f"/share/kuran/data/joseph/korean_corpus/ko-en-corpus.test.{target_lang}"], 
                       capture_output=True, text=True)
        
        ribes = subprocess.run(["sacrebleu", f"/share/kuran/data/joseph/korean_corpus/ko-en-corpus.test.{target_lang}", 
                       "-tok", "none", "-i", "out.tok", "-b"], capture_output=True, text=True)

    bleu_scores.append(bleu.stdout.strip())
    ribes_scores.append(ribes.stdout.strip())


# clean everything
def clean():
    subprocess.run(["rm", "out.tok"])
    subprocess.run(["rm", "out.bpe"])
    subprocess.run(["rm", "out.bpe.log"])

# remove all the files
def remove(bpe, shuffled_output_file_name):
    subprocess.run(["rm", bpe])
    subprocess.run(["rm", shuffled_output_file_name])


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Please enter the source lang, target lang, and s (suffled) or ns (nonshuffled)")

    else:
        source_lang = sys.argv[1]
        target_lang = sys.argv[2]
        option = sys.argv[3]

        source_file = f"/share/kuran/data/joseph/korean_corpus/ko-en-corpus.test.{source_lang}"

        if option == "s":
            model = f"/share/kuran/models/mt/sockeye3/joseph/shuffled_{source_lang}_{target_lang}_model"
        
        else:
            model = f"/share/kuran/models/mt/sockeye3/joseph/{source_lang}_{target_lang}_model"

        for val in shuffle_values:
            shuffled_output_file_name = f"shuffled_{val}_ko-en-corpus.test.{source_lang}"
            output_shuffled_file(val, source_file, shuffled_output_file_name)
            bpe = bpe_convert(shuffled_output_file_name, source_lang, val)
            translate(bpe, source_lang, target_lang, model)
            clean()
            remove(bpe, shuffled_output_file_name)

        if option == "s":
            file = f"shuffled_{source_lang}_{target_lang}_results.txt"

        else:
            file = f"{source_lang}_{target_lang}_results.txt"

        with open(file, 'w') as f:
            f.write("BLEU: ")
            f.write(' '.join(bleu_scores))
            f.write("\n")

            f.write("RIBES: ")
            f.write(' '.join(ribes_scores))
            f.write("\n")




