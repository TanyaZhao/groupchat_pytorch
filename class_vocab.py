class VocabClass(object):

    def __init__(self):
        self.idx2word = []
        self.word2idx = {}

    def add_word(self, word):
        if word not in self.word2idx:
            new_id = self.size()
            self.idx2word.append(word)
            self.word2idx[word] = new_id

    def has_key(self, word):
        return self.word2idx.has_key(word)

    def get_id(self, word):
        return self.word2idx.get(word)

    def get_word(self, w_id):
        return self.idx2word[w_id]

    def size(self):
        return len(self.idx2word)

    def save(self, path):
        with open(path, 'w') as f:
            for i, w in enumerate(self.idx2word):
                print >> f, str(i) + '\t' + w.encode('utf-8')

    @classmethod
    def load(cls, path):
        vocab = VocabClass()
        with open(path) as f:
            for line in f:
                w = line.strip().split('\t')[1].decode('utf-8')
                vocab.add_word(w)
        return vocab

