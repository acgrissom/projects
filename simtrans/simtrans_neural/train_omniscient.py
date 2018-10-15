DATA_DIR = './data'
MODEL_DIR = './models/'

#train and test test set.
input=DATA_DIR + '/test/de-omniscient.out'
from keras_wrapper.dataset import Dataset, saveDataset
from data_engine.prepare_data import keep_n_captions
ds = Dataset('newsttest2014-deen-omni', 'de-en-omni', silence=False)
ds.setOutput(input,
             'train',
             type='text',
             id='target_text',
             tokenization='tokenize_none',
             build_vocabulary=True,
             pad_on_batch=True,
             sample_weights=True,
             max_text_len=30,
             max_words=30000,
             min_occ=0)

ds.setOutput(MODEL_DIR + '/test/omniscient.en.out',
             'val',
             type='text',
             id='target_text',
             pad_on_batch=True,
             tokenization='tokenize_none',
             sample_weights=True,
             max_text_len=30,
             max_words=0)

ds.setInput(DATA_DIR + '/test/newstest2014-deen-src.de.txt',
            'train',
            type='text',
            id='source_text',
            pad_on_batch=True,
            tokenization='tokenize_none',
            build_vocabulary=True,
            fill='end',
            max_text_len=30,
            max_words=30000,
            min_occ=0)
ds.setInput(MODEL_DIR + '/test/newstest2014-deen-src.de.txt',
            'val',
            type='text',
            id='source_text',
            pad_on_batch=True,
            tokenization='tokenize_none',
            fill='end',
            max_text_len=30,
            min_occ=0)
