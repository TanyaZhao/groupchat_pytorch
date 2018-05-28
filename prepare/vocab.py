# encoding:utf-8

PAD = '<PAD>'
UNK = '<UNKNOWN>'

class Vocab(object):
    def __init__(self):
        self.i2w = []
        self.w2i = {}

    def add_word(self, word):
        if word not in self.w2i:
            new_id = self.size()
            self.i2w.append(word)
            self.w2i[word] = new_id

    def has_key(self, word):
        return self.w2i.has_key(word)

    def get_id(self, word):
        return self.w2i.get(word)

    def get_word(self, w_id):
        return self.i2w[w_id]

    def size(self):
        return len(self.i2w)

    def save(self, path):
        with open(path, 'w') as f:
            for i, w in enumerate(self.i2w):
                print >> f, str(i) + '\t' + w.encode('utf-8')

    @classmethod
    def load(cls, path):
        vocab = Vocab()
        with open(path) as f:
            for line in f:
                w = line.strip().split('\t')[1].decode('utf-8')
                vocab.add_word(w)
        return vocab



def doc_to_id(docs, vocab_word):
    """
    :param docs: 1D: n_docs, 2D: n_utters, 3D: (time, speaker_id, addressee_id, response, ..., label)
    :param vocab_word: Vocab()
    :return: docs: 1D: n_docs, 2D: n_utters, 3D: (time, speaker_id, addressee_id, response, ..., label)
    """

    if docs is None:
        return None
    count = 0
    for doc in docs:
        for sent in doc:
            spk_w2v_id = word_to_id(sent[1], vocab_word)
            sent[1] = spk_w2v_id
            sent[3] = sentence_to_id(tokens=sent[3], vocab_word=vocab_word)  # cand_response1
            sent[3].insert(0, spk_w2v_id)
            if sent[2] != '-':
                sent[2] = word_to_id(sent[2], vocab_word)
                for i, r in enumerate(sent[4:-1]): # cand_response2,...,label
                    sent[4 + i] = sentence_to_id(tokens=r, vocab_word=vocab_word)
                    sent[4 + i].insert(0, spk_w2v_id)
                count += 1

    print('\n\tQuestions: {:>8}'.format(count))
    return docs


def sentence_to_id(tokens, vocab_word):
    w_ids = []
    for token in tokens:
        if vocab_word.has_key(token):
            w_ids.append(vocab_word.get_id(token))
        else:
            w_ids.append(vocab_word.get_id(UNK))
    return w_ids


def word_to_id(token, vocab_word):
    if vocab_word.has_key(token):
        w_id = vocab_word.get_id(token)
    else:
        w_id = vocab_word.get_id(UNK)
    return w_id