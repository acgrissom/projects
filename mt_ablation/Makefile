anaconda.sh:
	curl https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh > anaconda.sh
	sh anaconda.sh
	rm anaconda.sh


apex:
	git clone https://github.com/NVIDIA/apex

apex_install: apex
	cd apex && pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

conda_environment:
	conda create -n sockeye python=3.8	

sockeye:
	pip install sockeye --no-deps
	pip install tensorboard
	pip install subword_nmt
	pip install packaging

sockeye_optional:
	git clone https://github.com/awslabs/sockeye
	cd sockeye && pip install .[optional]

genseqcopy.py:
	wget https://raw.githubusercontent.com/awslabs/sockeye/main/docs/tutorials/seqcopy/genseqcopy.py
	python3 genseqcopy.py

seqcopy: genseqcopy.py
	python3 -m sockeye.train -s data/train.source \
                         -t data/train.target \
                         -vs data/dev.source \
                         -vt data/dev.target \
                         --encoder transformer --decoder transformer \
                         --num-layers 1:1 \
                         --num-embed 32 \
                         --transformer-model-size 32 \
                         --transformer-feed-forward-num-hidden 64 \
                         --transformer-attention-heads 4 \
                         --max-num-checkpoint-not-improved 3 \
                         -o seqcopy_model


corpus.tc.en:
	wget http://data.statmt.org/wmt17/translation-task/preprocessed/de-en/corpus.tc.de.gz
	wget http://data.statmt.org/wmt17/translation-task/preprocessed/de-en/corpus.tc.en.gz
	gunzip corpus.tc.de.gz
	gunzip corpus.tc.en.gz
	curl https://data.statmt.org/wmt17/translation-task/preprocessed/de-en/dev.tgz | tar xvzf -

corpus.tc.de:
	wget http://data.statmt.org/wmt17/translation-task/preprocessed/de-en/corpus.tc.de.gz
	wget http://data.statmt.org/wmt17/translation-task/preprocessed/de-en/corpus.tc.en.gz
	gunzip corpus.tc.de.gz
	gunzip corpus.tc.en.gz
	wget http://data.statmt.org/wmt17/translation-task/preprocessed/de-en/dev.tgz
	tar xvzf dev.tgz


bpe.codes: corpus.tc.de
	subword-nmt learn-joint-bpe-and-vocab --input corpus.tc.de corpus.tc.en \
                                    -s 30000 \
                                    -o bpe.codes \
                                    --write-vocabulary bpe.vocab.de bpe.vocab.en


corpus.tc.BPE.de: bpe.codes
	subword-nmt apply-bpe -c bpe.codes --vocabulary bpe.vocab.de --vocabulary-threshold 50 < corpus.tc.de > corpus.tc.BPE.de

corpus.tc.BPE.en: bpe.codes
	subword-nmt apply-bpe -c bpe.codes --vocabulary bpe.vocab.en --vocabulary-threshold 50 < corpus.tc.en > corpus.tc.BPE.en


newstest2016.tc.BPE.de: corpus.tc.BPE.de
	subword-nmt apply-bpe -c bpe.codes --vocabulary bpe.vocab.de --vocabulary-threshold 50 < newstest2016.tc.de > newstest2016.tc.BPE.de

newstest2016.tc.BPE.en: corpus.tc.BPE.en
	subword-nmt apply-bpe -c bpe.codes --vocabulary bpe.vocab.en --vocabulary-threshold 50 < newstest2016.tc.en > newstest2016.tc.BPE.en

train_data: newstest2016.tc.BPE.en newstest2016.tc.BPE.de
	python -m sockeye.prepare_data \
                        -s corpus.tc.BPE.de \
                        -t corpus.tc.BPE.en \
                        -o train_data \
                        --shared-vocab

wmt_model: train_data
	python -m sockeye.train -d train_data \
                        -vs newstest2016.tc.BPE.de \
                        -vt newstest2016.tc.BPE.en \
                        --max-seq-len 60 \
                        --decode-and-evaluate 500 \
                        -o wmt_model \
                        --shared-vocab \
                        --max-num-epochs 3

clean:
	rm -Rf seqcopy_model
	rm -Rf sockeye
	rm -Rf apex
	rm -f anaconda.sh

clean_data:
	rm -Rf data
	rm -f corpus*.gz
	rm -f dev.tgz
	rm -Rf train_data
	rm -Rf wmt_model
	rm -RF *.sgm

# this corpus is quite small and does not produce good results. The new dataset is inside /share/kuran/data/joseph.
ko_corpus:
	curl -JLO https://github.com/jungyeul/korean-parallel-corpora/raw/master/korean-english-news-v1/korean-english-park.train.tar.gz
	curl -JLO https://github.com/jungyeul/korean-parallel-corpora/raw/master/korean-english-news-v1/korean-english-park.dev.tar.gz 
	curl -JLO https://github.com/jungyeul/korean-parallel-corpora/raw/master/korean-english-news-v1/korean-english-park.test.tar.gz

	tar xvzf korean-english-park.train.tar.gz
	tar xvzf korean-english-park.test.tar.gz
	tar xvzf korean-english-park.dev.tar.gz

# creating bpe files for ko-en-corpus
bpe_ko.codes:
	subword-nmt learn-joint-bpe-and-vocab --input /share/kuran/data/joseph/korean_corpus/ko-en-corpus.train.ko /share/kuran/data/joseph/korean_corpus/ko-en-corpus.train.en \
                                    -s 32000 \
                                    -o bpe_ko.codes \
                                    --write-vocabulary bpe.vocab.ko bpe.vocab.en

bpe.ko-en-corpus.train.ko:
	subword-nmt apply-bpe -c bpe_ko.codes --vocabulary bpe.vocab.ko <  /share/kuran/data/joseph/korean_corpus/ko-en-corpus.train.ko >  bpe.ko-en-corpus.train.ko

bpe.ko-en-corpus.train.en:
	subword-nmt apply-bpe -c bpe_ko.codes --vocabulary bpe.vocab.en <  /share/kuran/data/joseph/korean_corpus/ko-en-corpus.train.en >  bpe.ko-en-corpus.train.en

bpe.ko-en-corpus.val.ko:
	subword-nmt apply-bpe -c bpe_ko.codes --vocabulary bpe.vocab.ko <  /share/kuran/data/joseph/korean_corpus/ko-en-corpus.val.ko >  bpe.ko-en-corpus.val.ko

bpe.ko-en-corpus.val.en:
	subword-nmt apply-bpe -c bpe_ko.codes --vocabulary bpe.vocab.en <  /share/kuran/data/joseph/korean_corpus/ko-en-corpus.val.en >  bpe.ko-en-corpus.val.en

bpe.ko-en-corpus.test.ko:
	subword-nmt apply-bpe -c bpe_ko.codes --vocabulary bpe.vocab.ko <  /share/kuran/data/joseph/korean_corpus/ko-en-corpus.test.ko >  bpe.ko-en-corpus.test.ko

bpe.ko-en-corpus.test.en:
	subword-nmt apply-bpe -c bpe_ko.codes --vocabulary bpe.vocab.en <  /share/kuran/data/joseph/korean_corpus/ko-en-corpus.test.en >  bpe.ko-en-corpus.test.en

# apply bpe to all
bpe_all: bpe.ko-en-corpus.train.ko bpe.ko-en-corpus.train.en bpe.ko-en-corpus.val.ko bpe.ko-en-corpus.val.en bpe.ko-en-corpus.test.ko bpe.ko-en-corpus.test.en

morph.ko-en-corpus.train.ko:
	python morpheme.py /share/kuran/data/joseph/korean_corpus/ko-en-corpus.train.ko morph.ko-en-corpus.train.ko

morph/ko-en-corpus.test.ko:
	python morpheme.py /share/kuran/data/joseph/korean_corpus/ko-en-corpus.test.ko morph/ko-en-corpus.test.ko

morph/ko-en-corpus.val.ko:
	python morpheme.py /share/kuran/data/joseph/korean_corpus/ko-en-corpus.val.ko morph.ko-en-corpus.val.ko

# apply morph to all
morpheme_all: morph.ko-en-corpus.train.ko morph/ko-en-corpus.test.ko morph/ko-en-corpus.val.ko

ko_en_model:
	python train.py ko en ns

shuffled_ko_en_model:
	python train.py ko en s

en_ko_model:
	python train.py en ko ns

shuffled_en_ko_model:
	python train.py en ko s

shuffled_ko_ko_model:
	python train.py ko ko s

results_ko_en.txt:
	python shuffle_test.py ko en ns

results_shuffled_ko_en.txt:
	python shuffle_test.py ko en s

results_en_ko.txt:
	python shuffle_test.py en ko ns

results_shuffled_en_ko.txt:
	python shuffle_test.py en ko s

results_all: results_ko_en.txt results_shuffled_ko_en.txt results_en_ko.txt results_shuffled_en_ko.txt

val_en_ko.csv:
	python validation_bleu.py en ko

val_ko_en.csv:
	python validation_bleu.py ko en

percentage_shuffle.svg:
	python plot.py bleu

percentage_shuffle_RIBES.svg:
	python plot.py ribes

percentage_shuffle2.svg:
	python plot2.py bleu

percentage_shuffle2_RIBES.svg:
	python plot2.py ribes

plot_all: percentage_shuffle.svg percentage_shuffle_RIBES.svg percentage_shuffle2.svg percentage_shuffle2_RIBES.svg

bpe_clean:
	rm bpe*

results_clean:
	rm results*

plot_clean:
	rm percentage_shuffle*

clean_all: bpe_clean results_clean plot_clean


