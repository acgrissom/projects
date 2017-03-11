#!/usr/bin/python

references = ["Israeli officials are responsible for airport security".split(),
        "Israel is in charge of the security at this airport".split(),
        """The security work for this airport is the responsibility of the
        Israel government""".split(),
        "Israeli side was in charge of the security of this airport".split()]
system_a = "Israeli officials responsibility of airport safety".split()
system_b = "airport security Israeli officials are responsible".split()


cpdef list ngrams(list sentence, int n):
    """Returns the ngrams from a given sentence for a given n."""
    # assert isinstance(sentence, list), "Sentences are lists, got %s: %s" \
    #     % (str(type(sentence)), str(sentence))

    cdef list ngrams = []
    for start in range(0, len(sentence) - n + 1):
        ngrams.append(sentence[start:start + n])

    return ngrams


cpdef float brevity_penalty(list references, output):
    # Determine the length of the reference closest in length to the output
    assert len(references) > 0, "References empty!"
    cdef float reference_length = len(references[0])
    cdef float brevity_penalty
    cdef float output_len = float(len(output))
    #for ref in references:
    reference_length = min(1, output_len / reference_length)
    # reference_length = min([len(x) for x in references],
    #                        key= lambda y: y - len(output))
    brevity_penalty = min(1, output_len / reference_length)
    return brevity_penalty

cpdef float bleu(int N, list references, output, brevity=True):
    """Implementation of BLEU-N automatic evaluation metric, with lambda=1
    using multiple references."""
    cdef list precisions = []
    cdef int n
    cdef list output_ngrams
    cdef int relevant
    for n in range(1, N + 1):
        output_ngrams = ngrams(output, n)
        relevant = 0
        for ngram in output_ngrams:
            for reference in references:
                reference_ngrams = ngrams(reference, n)
                if ngram in reference_ngrams:
                    relevant += 1
                    reference_ngrams.remove(ngram)
                    break

        # If the output is too short, then we obviously didn't find anything
        # relevant
        if output_ngrams:
            precisions.append(float(relevant) / float(len(output_ngrams)))
        else:
            precisions.append(0.0)

    #product = reduce(lambda x, y: x * y, precisions)
    cdef float product = 1.0
    cdef float i
    for i in precisions:
        product *= i

    if brevity:
        return brevity_penalty(references, output) * product
    else:
        return product
