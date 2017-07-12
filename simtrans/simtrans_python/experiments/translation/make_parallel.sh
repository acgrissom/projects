#cdec/corpus/tokenize-anything.sh < ../../../../data/europarl-v7.en.2500.dev | cdec/corpus/lowercase.pl  > ../../../../data/europarl-v7.en.2500.dev.tok
cdec/corpus/paste-files.pl ../../../../data/europarl-v7.de.2500.dev ../../../../data/europarl-v7.en.2500.dev > ../../../../data/europarl-v7.en.2500.dev.parallel
