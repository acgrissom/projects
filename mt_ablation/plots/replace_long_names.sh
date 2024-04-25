sed -i 's/all named entities/NER-all/g' new.csv
sed -i 's/named organizations/NER-org/g' new.csv
sed -i 's/named people/NER-per/g' new.csv
sed -i 's/removed punctuation/punct./g' new.csv
sed -i 's/cardinal numbers/number/g' new.csv
sed -i 's/word corruption/corrupt/g' new.csv
sed -i 's/prepositions/prep./g' new.csv
sed -i 's/articles/article/g' new.csv
sed -i 's/lemmatization/lemma/g' new.csv
