
class ParallelInstance:
    def __init__(self, source_prefix, source_verb, target, id):
        self.src_pfx = source_prefix
        self.src_vb = source_verb
        self.tgt = target
        self.id = id

    def __len__(self):
        return len(self.src_pfx) + len(self.src_vb)

    def __str__(self):
        return "ID: %i SRC: %s | %s TGT: %s" % (self.id, " ".join(self.src_pfx),
                                                " ".join(self.src_vb), " ".join(self.tgt))

    # TODO(jbg): Update when we have multiple references per sentence
    def references(self):
        assert isinstance(self.tgt, list), \
            "Must be list: %s" % str(self.tgt)
        return [self.tgt]

