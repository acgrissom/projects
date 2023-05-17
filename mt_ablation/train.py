import random
import random
import sys
import subprocess
import os


# function to shuffle a sentence with given percentage
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

# function to create a new shuffled file
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

    f = open(shuffled_output_file_name, "w")
    for line in all_lines:
        f.write(line)
    f.close()

# function to train an unshuffled model
def train(source_lang, target_lang):

    subprocess.run(["python", "-m", "sockeye.prepare_data", "-s", 
                    f"bpe.ko-en-corpus.train.{source_lang}", "-t", f"bpe.ko-en-corpus.train.{target_lang}", 
                    "--shared-vocab", "--word-min-count",  "2", "--pad-vocab-to-multiple-of", 
                    "8", "--max-seq-len", "95", "--num-samples-per-shard", "10000000", 
                    "-o", f"{source_lang}_{target_lang}_prepare_data"])

    subprocess.run(["python", "-m", "sockeye.train", "-d", f"{source_lang}_{target_lang}_prepare_data",
                "--validation-source", f"bpe.ko-en-corpus.val.{source_lang}", 
                "--validation-target", f"bpe.ko-en-corpus.val.{target_lang}", 
                "--output", f"{source_lang}_{target_lang}_model", "--num-layers", "6", 
                "--transformer-model-size", "1024", "--transformer-attention-heads", "16", 
                "--transformer-feed-forward-num-hidden", "4096", "--amp", "--batch-type", "max-word", 
                "--batch-size", "5000", "--update-interval", "80", "--checkpoint-interval", "500", 
                "--max-num-checkpoint-not-improved", "20", "--optimizer-betas", "0.9:0.98", 
                "--initial-learning-rate", "0.06325", 
                "--learning-rate-scheduler-type", "inv-sqrt-decay", "--learning-rate-warmup", "4000", 
                "--seed", "1"])

def train_shuffle(bpe, shuffled_output_file_name, source_lang, target_lang):

    os.system(f"subword-nmt apply-bpe -c bpe_ko.codes --vocabulary  bpe.vocab.{source_lang} <  {shuffled_output_file_name} >  {bpe}")
    subprocess.run(["python", "-m", "sockeye.prepare_data", "-s", 
                    f"{bpe}", "-t", f"bpe.ko-en-corpus.train.{target_lang}", 
                    "--shared-vocab", "--word-min-count",  "2", "--pad-vocab-to-multiple-of", 
                    "8", "--max-seq-len", "95", "--num-samples-per-shard", "10000000", 
                    "-o", f"shuffled_{source_lang}_{target_lang}_prepare_data"])

    subprocess.run(["python", "-m", "sockeye.train", "-d", f"shuffled_{source_lang}_{target_lang}_prepare_data", 
                "--validation-source", f"bpe.ko-en-corpus.val.{source_lang}", 
                "--validation-target", f"bpe.ko-en-corpus.val.{target_lang}", 
                "--output", f"shuffled_{source_lang}_{target_lang}_model", "--num-layers", "6", 
                "--transformer-model-size", "1024", "--transformer-attention-heads", "16", 
                "--transformer-feed-forward-num-hidden", "4096", "--amp", "--batch-type", "max-word", 
                "--batch-size", "5000", "--update-interval", "80", "--checkpoint-interval", "500", 
                "--max-num-checkpoint-not-improved", "20", "--optimizer-betas", "0.9:0.98", 
                "--initial-learning-rate", "0.06325", 
                "--learning-rate-scheduler-type", "inv-sqrt-decay", "--learning-rate-warmup", "4000", 
                "--seed", "1"])

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Please enter the source lang, target lang, and s (shuffled) or ns (nonshuffled)")
    else:
        source_lang = sys.argv[1]
        target_lang = sys.argv[2]
        option = sys.argv[3]

        source_file = f"/share/kuran/data/joseph/korean_corpus/ko-en-corpus.train.{source_lang}"
        shuffled_output_file_name = f"shuffled.ko-en-corpus.train.{source_lang}"
        bpe = f"bpe.shuffled.ko-en-corpus.train.{source_lang}"

        if option == "s":
            output_shuffled_file(1.0, source_file, shuffled_output_file_name)
            train_shuffle(bpe, shuffled_output_file_name, source_lang, target_lang)

            os.system(f"rm {shuffled_output_file_name}")
            os.system(f"rm {bpe}")

        else:
            train(source_lang, target_lang)
