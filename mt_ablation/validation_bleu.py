from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import sys
import pandas as pd
import subprocess
import os
import math

# translating the sentences from validation corpus
def translate(source_lang, target_lang, original_bpe):

    subprocess.run(["python", "-m", "sockeye.translate", "--input", original_bpe,"--output", f"bpe.ko-en-nonshuffle.{target_lang}", 
                    "--model", f"/share/kuran/models/mt/sockeye3/joseph/{source_lang}_{target_lang}_model", "--dtype", "float16", 
                    "--beam-size", "5","--batch-size", "64"])

    os.system(f"sed -re 's/(@@ |@@$)//g' < bpe.ko-en-nonshuffle.{target_lang} > ko-en-nonshuffle.{target_lang}")
    os.system(f"rm bpe.ko-en-nonshuffle.{target_lang}")

    subprocess.run(["python", "-m", "sockeye.translate", "--input", original_bpe,"--output", f"bpe.ko-en-shuffle.{target_lang}", 
                    "--model", f"/share/kuran/models/mt/sockeye3/joseph/shuffled_{source_lang}_{target_lang}_model", "--dtype", 
                    "float16", "--beam-size", "5","--batch-size", "64"])

    os.system(f"sed -re 's/(@@ |@@$)//g' < bpe.ko-en-shuffle.{target_lang} > ko-en-shuffle.{target_lang}")
    os.system(f"rm bpe.ko-en-shuffle.{target_lang}")


# creating a csv file of all the lines
def create_csv(original, reference, nonshuffle, shuffle, source_lang, target_lang):
    smoothie = SmoothingFunction().method1
    df = pd.DataFrame(columns=['original', 'reference', 'nonshuffle', 'shuffle', 's_ns_difference', 
                               'nonshuffle_correct', 'shuffle_correct', 'order change', 'structural change', 
                               'extra words', 'missing words', 'incorrect words', 'punctuation', 'extraneous errors', 'comment'])

    with open(original, 'r') as f1, \
         open(reference, 'r') as f2, \
         open(nonshuffle, 'r') as f3, \
         open(shuffle, 'r') as f4:
        
        original_sen = f1.readlines()
        reference_sen = f2.readlines()
        nonshuffle_sen = f3.readlines()
        shuffle_sen = f4.readlines()

    for i in range(len(original_sen)):

        if target_lang == "en":
            ns_score = sentence_bleu([reference_sen[i].strip().lower().split()], nonshuffle_sen[i].strip().lower().split(), 
                                 smoothing_function=smoothie)
        
            s_score = sentence_bleu([reference_sen[i].strip().lower().split()], shuffle_sen[i].strip().lower().split(), 
                                 smoothing_function=smoothie)
            
        else:
            ns_score = sentence_bleu([reference_sen[i].strip().split()], nonshuffle_sen[i].strip().split(), 
                                 smoothing_function=smoothie)
        
            s_score = sentence_bleu([reference_sen[i].strip().split()], shuffle_sen[i].strip().split(), 
                                 smoothing_function=smoothie)
 
        df.loc[len(df)] = [original_sen[i].strip(), reference_sen[i].strip(), nonshuffle_sen[i].strip(), 
                           shuffle_sen[i].strip(), s_score - ns_score, 0, 1, 0, 0, 0, 0, 0, 0, 0, " "]

    return df


def calculate(csv_name, source_lang, target_lang):
    df = pd.read_csv(csv_name)
    nonshuffle_correct = 0
    shuffle_correct = 0
    order = 0
    structural = 0
    missing = 0
    extra = 0
    incorrect = 0
    punctuation = 0
    extraneous = 0
    comments = {}

    for i in range(100):
        if df.loc[i, 'nonshuffle_correct'] == 1: nonshuffle_correct += 1
        if df.loc[i, 'shuffle_correct'] == 1: shuffle_correct += 1
        if df.loc[i, 'order change'] == 1: order += 1
        if df.loc[i, 'structural change'] == 1: structural += 1
        if df.loc[i, 'extra words'] == 1: extra += 1
        if df.loc[i, 'missing words'] == 1: missing += 1
        if df.loc[i, 'incorrect words'] == 1: incorrect += 1
        if df.loc[i, 'punctuation'] == 1: punctuation += 1
        if df.loc[i, 'extraneous errors'] == 1: extraneous += 1    

        if type(df.loc[i, 'comment']) == str:
            comments[i] = df.loc[i, 'comment']

    out_file = f"val_{source_lang}_{target_lang}_labelled.txt"

    with open(out_file, 'w') as f:
        f.write(f"Out of 100 setences:\n")
        f.write(f"Nonshuffle Correct: {nonshuffle_correct}\n")
        f.write(f"Shuffle Correct: {shuffle_correct}\n")
        f.write(f"Order change: {order}\n")
        f.write(f"Structural change: {structural}\n")
        f.write(f"Extra words: {extra}\n")
        f.write(f"Missing words: {missing}\n")
        f.write(f"Incorrect words: {incorrect}\n")
        f.write(f"Punctuation: {punctuation}\n")
        f.write(f"Extraneous errors: {extraneous}\n")

        f.write("\nLines that had comments:\n")
        for i in comments.keys():
            f.write(f"{i}) {comments[i]}\n")

def add(csv_name):
    df = pd.read_csv(csv_name, index_col=False)

    zeros = [0] * 100
    ones = [1] * 100

    #df = df.drop("punctuation", axis=1)
    df.insert(df.shape[1] -2, "punctuation", zeros)
    df.insert(6, "shuffle_correct", ones)
    df = df.rename(columns={"correct": "nonshuffle_correct"})

    df.to_csv(csv_name, index=False)


def clean(nonshuffle, shuffle):
    subprocess.run(["rm", nonshuffle])
    subprocess.run(['rm', shuffle])


if __name__ == "__main__":
    source_lang = sys.argv[1]
    target_lang = sys.argv[2]

    original = f"/share/kuran/data/joseph/korean_corpus/ko-en-corpus.val.{source_lang}"
    original_bpe = f"bpe.ko-en-corpus.val.{source_lang}"
    reference = f"/share/kuran/data/joseph/korean_corpus/ko-en-corpus.val.{target_lang}"

    nonshuffle = f"ko-en-nonshuffle.{target_lang}"
    shuffle = f"ko-en-shuffle.{target_lang}"

    translate(source_lang, target_lang, original_bpe)
    df = create_csv(original, reference, nonshuffle, shuffle, source_lang, target_lang)

    df_sorted = df.sort_values(by="s_ns_difference", ascending=False)
    edge = pd.concat([df_sorted[:100]])
    edge.to_csv(f"val_{source_lang}_{target_lang}.csv", index=False)

    calculate(f"val_{source_lang}_{target_lang}_labelled.csv", source_lang, target_lang)
    clean(nonshuffle, shuffle)
