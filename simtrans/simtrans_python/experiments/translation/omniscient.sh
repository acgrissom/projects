plain2snt.out ../../../../../../data/europarl-v7.en.2500.dev_and_test  ../../../../../../data/europarl-v7.de.2500.dev_and_test
snt2cooc.out  ../../../../../../data/europarl-v7.de.2500.dev_and_test.vcb ../../../../../../data/europarl-v7.en.2500.dev_and_test.vcb ../../../../../../data/europarl-v7.en.2500.dev_and_test_europarl-v7.de.2500.dev_and_test.snt > ../scratch/europarl-de-en.cooc
GIZA++ -s ../../../../../../data/europarl-v7.de.2500.dev_and_test.vcb -t ../../../../../../data/europarl-v7.en.2500.dev_and_test.vcb -c ../../../../../../data/europarl-v7.en.2500.dev_and_test_europarl-v7.de.2500.dev_and_test.snt  -CoocurrenceFile\
 ../scratch/europarl-de-en.cooc -o ../scratch/mt -nbestalignments 1 -model4iterations 0
