mkdir data
wget http://www.statmt.org/wmt13/training-monolingual-europarl-v7.tgz
wget http://www.statmt.org/wmt14/dev.tgz
wget http://www.statmt.org/wmt14/test-full.tgz

mv training-monolingual-europarl-v7.tgz data/
mv dev.tgz data/
mv test-full.tgz data/

cd data

tar xvf *.tgz
rm training-monolingual-europarl-v7.tgz
rm dev.tgz
rm test-full.tgz

cd ..

python3 sgm2txt.py < data/test/newstest2014-deen-ref.en.sgm > data/test/newstest2014-deen-ref.en.txt
python3 sgm2txt.py < data/test/newstest2014-deen-ref.de.sgm > data/test/newstest2014-deen-ref.de.txt
