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
* `train.py`: trains both shuffled and unshuffled models with one language as source and the other as target
* `shuffle_test.py`: shuffles the source test ccorpus, uses it for translation for both shuffled and unshuffled models, and calculates the RIBES and BLEU score.

To run the whole process of Korean English MT-ablation:
```bash 
make ko-everything 
```
