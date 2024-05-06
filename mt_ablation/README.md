### MT ablation-related projects

#### Things to run before to set up environment

To run demo pipeline:

1.  Install anaconda

```bash
make anaconda.sh
```

2. Create sockeye environment.
```bash
make conda_environment
conda activate sockeye
make sockeye
make sockeye_optional
```

3. Install apex

```bash
make apex_install
```

#### MT demo pipelines from Sockeye web site.

Sequence copy demo:
```bash
make genseqcopy.py
```

(Longer) WMT2014  Demo:
```bash
make wmt_demo
```

#### Files and Makefile Targets for Ko-En-Corpus

Python files:

* `morpheme.py`: converts a korean corpus into a new corpus separated by morphemes
* `ribes_score.py`: calculates the ribes score of a translated test corpus
* `ribes_Score_ko.py`: `ribes_score.py` but for korean as the target language
* `train.py`: trains both shuffled and unshuffled models with one language as source and the other as target
* `shuffle_test.py`: shuffles the source test ccorpus, uses it for translation for both shuffled and unshuffled models, and calculates the RIBES and BLEU score.
* `plot.py`: plots scores of en-fr, en-es, and es-fr in RIBES and BLEU
* `plot2.py`: plots scores of en-ko.
* `validation_bleu.py`: creates a csv file of top 100 sentences where shuffle scored better than nonshuffle.

Other files:

* `val_en_ko_labelled.csv`: top 100 sentences where shuffled scored better than nonshuffled for en-ko models with labels to show types of errors
* `val_en_ko_labelled.txt`: reads through `val_en_ko_labelled.csv` and counts the different errors along with comments
* `val_ko_en_labelled.csv`: top 100 sentences where shuffled scored better than nonshuffled for ko-en models with labels to show types of errors
* `val_ko_en_labelled.txt`: reads through `val_ko_en_labelled.csv` and counts the different errors along with comments

To run the whole process of Korean English MT-ablation:

1. Tokenize korean corpus with BPE

```bash 
make bpe_ko.codes
make bpe_all
```

2. Produce BLEU and RIBES score for all models

```bash
make results_all
```

3. Plotting the scores in svg

```bash
make plot_all
```

4. Deleting files that were created

```bash
make bpe_clean
make results_clean
make plot_clean
```

Optional: Train models (optional as the models are already backed up, this process takes a lot of time)

```bash
make ko_en_model
make shuffled_ko_en_model
make en_ko_model
make shuffled_en_ko_model
make shuffled_ko_ko_model
```


