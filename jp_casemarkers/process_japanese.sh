#!/bin/bash

input_file="$1"
# stores mecab output
output_file="${input_file%.*}_mecab.txt"
jp_file="${input_file%.*}_no_particles.jp"


# Run MeCab to generate output file
mecab "$input_file" -o "$output_file"

# Run Python script on the output file
python3 test.py "$output_file" "$jp_file"


