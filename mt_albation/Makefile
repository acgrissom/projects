anaconda.sh:
	curl https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh > anaconda.sh
	sh anaconda.sh
	rm anaconda.sh

conda_environment:
	conda create --name sockeye
	conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y


apex:
	git clone https://github.com/NVIDIA/apex

apex_install: apex
	cd apex && pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

sockeye:
	conda create -n sockeye python=3.8
	source activate sockeye
	pip install sockeye --no-deps
	pip install tensorboard
	pip install subword_nmt

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
	curl http://data.statmt.org/wmt17/translation-task/preprocessed/de-en/dev.tgz | tar xvzf -

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
