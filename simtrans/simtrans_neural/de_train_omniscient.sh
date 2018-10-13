#rm f models/newstest2014-deen-ref.de.
#Train
DATA_DIR=data
MODEL_DIR=models/deen/omniscient
mkdir -p $MODEL_DIR
python3 -m sockeye.train\
                       --source $DATA_DIR/test/newstest2014-deen-src.de.txt \
                       --target $DATA_DIR/test/newstest2014-deen-ref.en.txt \
                       --encoder transformer \
                       --decoder transformer \
                       --validation-source $DATA_DIR/test/newstest2014-deen-src.de.txt \
                       --validation-target $DATA_DIR/test/newstest2014-deen-ref.en.txt \
                       --output $MODEL_DIR \
		       --decode-and-evaluate 500 
