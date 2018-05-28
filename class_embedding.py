# encoding:utf-8
import gzip
from class_vocab import VocabClass
import numpy as np

PAD = '<PAD>'
UNK = '<UNKNOWN>'

class EmbeddingClass(object):
    def __init__(self, emb_size, agent_emb_size=0, init_emb=None):
        self.emb_size = emb_size
        self.agent_emb_size = agent_emb_size
        self.init_emb = init_emb


    def get_embedding(self, raw_train, raw_dev, raw_test):
        # get vocab
        train_dataset, vocab = self.get_dataset(raw_train)
        dev_dataset, vocab = self.get_dataset(raw_dev, vocab)
        test_dataset, vocab = self.get_dataset(raw_test, vocab)

        print "vocab size {}".format(len(vocab))

        vocab_class = VocabClass()
        vocab_class.add_word(PAD)
        vocab_class.add_word(UNK)

        # word embedding
        if self.init_emb is None:
            for w in vocab:
                vocab_class.add_word(w)
            print('\n\tRandom Initialized Word Embeddings')
            emb = np.random.rand(vocab_class.size(), self.emb_size)
        else:
            emb = []
            with open(self.init_emb) as lines:
                for line in lines:
                    line = line.strip().decode('utf-8').split()
                    w = line[0]
                    e = line[1:]

                    if not vocab_class.has_key(w):
                        vocab_class.add_word(w)
                        emb.append(e)

            np_emb = np.asarray(emb, dtype=np.float32)  # (n_words,emb)
            unk = np.mean(np_emb, 0)
            emb.insert(0, [0] * self.emb_size)  # insert pad
            emb.insert(1, unk.tolist())  # insert unk
            emb = np.asarray(emb, dtype=np.float32)

        # # agent embedding
        # for agent in agent_vocab:
        #     agent_class.add_word(agent)
        # print('\n\tRandom Initialized Agent Embeddings')
        # agent_emb = np.random.rand(agent_class.size(), self.agent_emb_size)
        #
        # np_emb = np.asarray(agent_emb, dtype=np.float32)  # (n_words,emb)
        # unk = np.mean(np_emb, 0)
        # agent_emb.insert(0, [0] * self.emb_size)  # insert pad
        # agent_emb.insert(1, unk.tolist())  # insert unk
        # agent_emb = np.asarray(agent_emb, dtype=np.float32)

        return vocab_class,emb


    def get_dataset(self, file_path, vocab=set([])):

        docs = []
        utterances = []
        file_open = gzip.open if file_path.endswith(".gz") else open

        with file_open(file_path) as gf:
            # line: (time, speaker_id, addressee_id, cand_res1, cand_res2, ... , label)
            for line in gf:
                line = line.rstrip().split("\t")

                if len(line) < 6:  # end of a thread
                    docs.append(utterances)
                    utterances = []
                else:
                    for i, sent in enumerate(line[3:-1]):  # cand_res1,cand_res2,label
                        words = []
                        vocab.add(line[1])
                        for w in sent.split():  # convert every word in cand_res to lowercase
                            w = w.lower()
                            vocab.add(w)
                            words.append(w)
                        line[3 + i] = words

                    line[-1] = -1 if line[-1] == '-' else int(line[-1])
                    utterances.append(line)

        return docs, vocab


    def get_dataset_agent(self, file_path, vocab=set([]), agent_vocab=set([])):

        docs = []
        utterances = []
        file_open = gzip.open if file_path.endswith(".gz") else open

        with file_open(file_path) as gf:
            # line: (time, speaker_id, addressee_id, cand_res1, cand_res2, ... , label)
            for line in gf:
                line = line.rstrip().split("\t")

                if len(line) < 6:  # end of a thread
                    docs.append(utterances)
                    utterances = []
                else:
                    for i, sent in enumerate(line[3:-1]):  # cand_res1,cand_res2,label
                        words = []
                        agent_vocab.add(line[1])  # add agent
                        for w in sent.split():  # convert every word in cand_res to lowercase
                            w = w.lower()
                            vocab.add(w)
                            words.append(w)
                        line[3 + i] = words

                    line[-1] = -1 if line[-1] == '-' else int(line[-1])
                    utterances.append(line)

        return docs, vocab, agent_vocab



