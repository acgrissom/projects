import random
import random
import sys
from pathlib import Path
import subprocess
import os

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

#shuffle_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# shuffle_values = [0.2, 0.4, 0.6, 0.8]
shuffle_values = [1.0]

bleu_scores = []

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

    f = open(shuffled_output_file_name, "a")
    for line in all_lines:
        f.write(line)
    f.close()

# run sockeye-translate

def train(val, shuffled_output_file_name, source_lang, target_lang):
    os.system("subword-nmt apply-bpe -c bpe_ko.codes --vocabulary  bpe.vocab.ko <  {shuffled_output_file_name} >  {shuffled_output_file_name}.bpe")
    subprocess.run(["python", "-m", "sockeye.prepare_data", "-s", 
                    f"{shuffled_output_file_name}.bpe", "-t", f"ko-en-corpus.train.BPE.{target_lang}", 
                    "--shared-vocab", "--word-min-count",  "2", "--pad-vocab-to-multiple-of", 
                    "8", "--max-seq-len", "95", "--num-samples-per-shard", "10000000", 
                    "-o", "ko_prepare_data"])

    subprocess.run(["python", "-m", "sockeye.train", "-d", "ko_prepare_data",
                "--validation-source", f"ko-en-corpus.val.BPE.{source_lang}", 
                "--validation-target", f"ko-en-corpus.val.BPE.{target_lang}", 
                "--output", f"{val}_ko_en_model", "--num-layers", "6", 
                "--transformer-model-size", "1024", "--transformer-attention-heads", "16", 
                "--transformer-feed-forward-num-hidden", "4096", "--amp", "--batch-type", "max-word", 
                "--batch-size", "5000", "--update-interval", "80", "--checkpoint-interval", "500", 
                "--max-num-checkpoint-not-improved", "20", "--optimizer-betas", "0.9:0.98", 
                "--initial-learning-rate", "0.06325", 
                "--learning-rate-scheduler-type", "inv-sqrt-decay", "--learning-rate-warmup 4000", 
                "--seed 1"])

def translate(val, source_lang):
    subprocess.run(["python", "-m", "sockeye-translate", "--input", f"ko-en-corpus.test.BPE.{source_lang}","--output", "out.bpe","--model", f"{val}_ko_en_model", "--dtype", "float16", "--beam-size", "5","--batch-size", "64"])


def evaluate(val, target_lang):
    os.system("sed -re 's/(@@ |@@$)//g' <out.bpe >out.tok")
    os.system(f"sacrebleu ko-en-corpus.test.{target_lang} -tok none -i out.tok > {val}_bleu.txt")
    os.ststem("python ribes_score.py ko-en-corpus.test.{target_lang}")

def clean(val)

    subprocess.run(["rm", "out.tok"])

    subprocess.run(["rm", "out.bpe"])

    subprocess.run(["rm", "ko_prepare_data"])

    subprocess.run(["mv", f"{val}_ko_en_model", "models/"])

    subprocess.run(["mv", f"{val}_bleu.txt", "models/")

    subprocess.run(["mv", "results_ribes.txt", f"{val}_ribes.txt"])

    subprocess.run(["mv", "{val_ribes.txt", "models/"])


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Please enter the name of the file to shuffle, the source lang, and the target lang")
    else:
        source_lang = sys.argv[1]
        target_lang = sys.argv[2]
        source_file = f"ko-en-corpus.train.{source_lang}"

        for val in shuffle_values:
            shuffled_output_file_name = f"{val}_shuffled.train.{source_lang}"
            output_shuffled_file(val, source_file, shuffled_output_file_name)
            train(val, shuffled_output_file_name, source_lang, target_lang)
            translate(val, source_lang)
            evaluate(val, target_lang)
            clean(val)
