
def generate_simple_sentences(input_nouns_singular:list, input_nouns_plural:list, input_adjectives:list):
    '''
    Generates a txt file with non-negated singular sentences in the form of 'a [noun] that is [adjective]'
    Generates a txt file with the negated counterparts of these singular sentences in the form of 'a [noun] that is not [adjective]'
    Generates a txt file with non-negated plural sentences in the form of '[plural noun] that are [adjective]'
    Generates a txt file with the negated counterparts of these plural sentences in the form of 'a [plural noun] that are not [adjective]'
    '''

    outputs_singular_non_negated = open('non_negated_sentences_singular.txt', 'w')
    outputs_singular_negated = open('negated_sentences_singular.txt', 'w')
    outputs_plural_non_negated = open('non_negated_sentences_plural.txt', 'w')
    outputs_plural_negated = open('negated_sentences_plural.txt', 'w')

    #singular nouns
    for sing_noun in input_nouns_singular:
        for sing_adj in input_adjectives:
            sing_sentence = 'a ' + sing_noun.strip() + ' that is ' + sing_adj.strip() #strip() to remove newlines
            sing_negated_sentence = 'a ' + sing_noun.strip() + ' that is not ' + sing_adj.strip() 
            outputs_singular_non_negated.write(sing_sentence + '\n')
            outputs_singular_negated.write(sing_negated_sentence + '\n')

    #plural nouns
    #TODO use .lower()?
    for plural_noun in input_nouns_plural:
        for plural_adj in input_adjectives:
            plural_sentence = plural_noun.strip() + ' that are ' + plural_adj.strip() #strip() to remove newlines
            plural_negated_sentence = plural_noun.strip() + ' that are not ' + plural_adj.strip() 
            outputs_plural_non_negated.write(plural_sentence + '\n')
            outputs_plural_negated.write(plural_negated_sentence + '\n')

def main():
    #keep the examples to a number where you can classify them all
    singular_nouns = open('singular_nouns.txt', 'r')
    plural_nouns = open('plural_nouns.txt', 'r')
    adjectives = open('adjectives.txt', 'r')

    singular_nouns_list = singular_nouns.readlines()
    plural_nouns_list = plural_nouns.readlines()
    adjectives_list = adjectives.readlines()

    generate_simple_sentences(singular_nouns_list, plural_nouns_list, adjectives_list)

main()

#Evaluation metrics: singular/plural, negated/non-negated, how well does it match the prompt (1-5), does it contradict the prompt (yes or no)

