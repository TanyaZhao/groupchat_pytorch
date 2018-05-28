# encoding:utf-8
import gzip
from vocab import Vocab, PAD, UNK, doc_to_id
import numpy as np
import argparse
from utils import get_samples, tf_samples
import json

def  get_datasets(argv, test=False):
    print('\nLoad dataset...')
    dataset = []
    if test:
        test_docs, vocab = load_dataset(fn=argv.test_data, data_size=argv.data_size, test=True)
        dataset.append(test_docs)
        return dataset, vocab
    else:
        train_docs, vocab = load_dataset(fn=argv.train_data, data_size=argv.data_size)
        dev_docs, vocab = load_dataset(fn=argv.dev_data, vocab=vocab, data_size=argv.data_size)
        dataset.append(train_docs)
        dataset.append(dev_docs)
        return dataset, vocab



def load_dataset(fn, vocab=set([]), data_size=1000000, test=False):
    """
    :param fn: file name
    :param vocab: vocab set
    :param data_size: how many threads are used
    :return: docs: 1D: n_doc, 2D: n_utterances, 3D: elem=(time, speaker_id, addressee_id, cand_res1, ... , label)
    """
    if fn is None:
        return None, vocab

    docs = []
    utterances = []
    file_open = gzip.open if fn.endswith(".gz") else open

    with file_open(fn) as gf:
        # line: (time, speaker_id, addressee_id, cand_res1, cand_res2, ... , label)
        for line in gf:
            line = line.rstrip().split("\t")

            if len(line) < 6:  # end of a thread
                docs.append(utterances)
                utterances = []

                if len(docs) >= data_size:
                    break
            else:
                for i, sent in enumerate(line[3:-1]):  # cand_res1,cand_res2,label
                    words = []
                    if test is False:
                        vocab.add(line[1])  # add speaker_id
                    for w in sent.split():  # convert every word in cand_res to lowercase
                        w = w.lower()
                        if test is False:  # if is train
                            vocab.add(w)
                        words.append(w)

                    line[3 + i] = words

                ##################
                # Label          #
                # -1: Not sample #
                # 0-: Sample     #
                ##################
                line[-1] = -1 if line[-1] == '-' else int(line[-1])
                utterances.append(line)

    return docs, vocab


def load_init_emb(init_emb, dim_emb, word_set=None):
    """
    :param init_emb: Column 0 = word, Column 1- = value; e.g., [the 0.418 0.24968 -0.41242 ...]
    :param word_set: the set of words that appear in train/dev/test set
    :return: vocab_word: Vocab()
    :return: emb: np.array
    """

    print('\nLoad initial word embedding...')

    vocab_word = Vocab()
    vocab_word.add_word(PAD)
    vocab_word.add_word(UNK)

    ################################
    # Load pre-trained  embeddings #
    ################################
    if init_emb is None:
        for w in word_set:
            vocab_word.add_word(w)
        emb = np.random.rand(vocab_word.size(), dim_emb)
        print('\n\tRandom Initialized Word Embeddings')
    else:
        emb = []
        with open(init_emb) as lines:
            for line in lines:
                line = line.strip().decode('utf-8').split()
                w = line[0]
                e = line[1:]
                dim_emb = len(e)

                if dim_emb == 300 or dim_emb == 50 or dim_emb == 100 or dim_emb == 200:
                    if not vocab_word.has_key(w):
                        vocab_word.add_word(w)
                        emb.append(e)

        np_emb = np.asarray(emb, dtype=np.float32)  # (n_words,emb)
        unk = np.mean(np_emb, 0)
        emb.insert(1, unk.tolist())  # unk
        emb.insert(0, [0] * dim_emb)  # pad
        emb = np.asarray(emb, dtype=np.float32)

        assert len(emb) == vocab_word.size(), 'emb: %d  vocab: %d' % (len(emb), vocab_word.size())
        print('\n\tWord Embedding Size: %d' % len(emb))

    return vocab_word, emb


def create_samples(argv, dataset, vocab_word, test):
    n_prev_sents = argv.n_prev_sents  # number of previous sentences, default:5
    max_n_words = argv.max_n_words  # default:20

    cands = dataset[0][0][3:-1]  # [cand_res1, cand_res2, label]
    n_cands = len(cands)

    print('\n\nTASK  SETTING')
    print('\n\tn_cands:%d  n_prev_sents:%d  max_n_words:%d\n' % (n_cands, n_prev_sents, max_n_words))


    # print('\n\nConverting words into ids...')
    # # samples: 1D: n_docs, 2D: n_utterances, 3D: (time, speaker_id, addressee_id, response, ..., label)
    # samples = doc_to_id(dataset, vocab_word)

    print('\n\nCreating samples...')
    # samples: 1D: n_samples; 2D: Sample
    samples = get_samples(dataset=dataset, n_prev_sents=n_prev_sents, max_n_words=max_n_words, pad=False)

    print "num of samples: %d" % len(samples)

    return samples


def get_embeddings(emb, dim_emb):
    parser = argparse.ArgumentParser(description='GroupChat')
    parser.add_argument('--mode', default='train', help='')
    # parser.add_argument('--train_data', default='data/input/train-data.cand-2.gz', help='path to a training data')
    # parser.add_argument('--dev_data', default='data/input/dev-data.cand-2.gz', help='path to a development data')
    # parser.add_argument('--test_data', default='data/input/dev-data-test', help='path to a test data')

    parser.add_argument('--train_data', default='data/input/train-data-test', help='path to a training data')
    parser.add_argument('--dev_data', default='data/input/dev-data-test', help='path to a development data')
    parser.add_argument('--test_data', default='data/input/dev-data-test', help='path to a test data')

    # parser.add_argument('--init_emb', default='data/w2v/groupchat_w2v_100d.txt', action='store',
    #                     help='Initial embedding')
    parser.add_argument('--init_emb', default=None, action='store', help='Initial embedding')
    parser.add_argument('--data_size', type=int, default=100000, help='number of docs used for the task')
    parser.add_argument('--response_size', type=int, default=2, help='number of candidate response')
    parser.add_argument('--sample_size', type=int, default=1,
                        help='number of division of samples used for the task')  # sample
    parser.add_argument('--n_cands', type=int, default=2, help='number of candidate responses')
    parser.add_argument('--n_prev_sents', type=int, default=5, help='number of previous sentences')
    parser.add_argument('--max_n_words', type=int, default=20, help='maximum number of words for context/response')
    parser.add_argument('--dim_emb', type=int, default=100, help='maximum number of words for context/response')

    print "Load embeddings ..."
    argv = parser.parse_args()
    dataset, vocab = get_datasets(argv)
    print "vocab size: "
    print len(vocab)

    vocab_words, embeddings = load_init_emb(emb, dim_emb, vocab)

    return vocab_words, embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GroupChat')
    parser.add_argument('--mode', default='train', help='')
    parser.add_argument('--train_data', default='../data/input/train-data.cand-2.gz', help='path to a training data')
    parser.add_argument('--dev_data', default='../data/input/dev-data.cand-2.gz', help='path to a development data')
    parser.add_argument('--test_data', default='../data/input/test-data.cand-2.gz', help='path to a test data')

    # parser.add_argument('--train_data', default='../data/input/train-data-test', help='path to a training data')
    # parser.add_argument('--dev_data', default='../data/input/dev-data-test', help='path to a development data')
    # parser.add_argument('--test_data', default='../data/input/test-data-test', help='path to a test data')

    # parser.add_argument('--init_emb', default='../data/w2v/groupchat_w2v_100d.txt', action='store',
    #                     help='Initial embedding')
    parser.add_argument('--init_emb', default=None, action='store', help='Initial embedding')
    parser.add_argument('--data_size', type=int, default=100000, help='number of docs used for the task')
    parser.add_argument('--response_size', type=int, default=2, help='number of candidate response')
    parser.add_argument('--sample_size', type=int, default=1,
                        help='number of division of samples used for the task')  # sample
    parser.add_argument('--n_cands', type=int, default=2, help='number of candidate responses')
    parser.add_argument('--n_prev_sents', type=int, default=5, help='number of previous sentences')
    parser.add_argument('--max_n_words', type=int, default=20, help='maximum number of words for context/response')
    parser.add_argument('--dim_emb', type=int, default=100, help='maximum number of words for context/response')

    argv = parser.parse_args()
    dataset, vocab = get_datasets(argv)
    vocab_words, embeddings = load_init_emb(argv.init_emb, argv.dim_emb, vocab)

    train_sampleset = create_samples(argv, dataset[0], vocab_words, test=False)
    with open('train_cand_2.txt', 'a') as fw:
        for sample in train_sampleset:
            sample_dict = {}
            sample_dict['context'] = sample.context
            sample_dict['context'] = sample.context
            sample_dict['context'] = sample.context
            sample_dict['response'] = sample.response
            sample_dict['spk_agents'] = sample.spk_agents
            sample_dict['true_adr'] = sample.true_adr
            sample_dict['true_res'] = sample.true_res
            # sample_dict['agent_index_dict'] = sample.agent_index_dict
            sample_json = json.dumps(sample_dict)

            fw.write(sample_json)
            fw.write('\n')
    fw.close()

    print 'train data finished.'

    print
    dev_sampleset = create_samples(argv, dataset[1], vocab_words, test=False)
    with open('dev_cand_2.txt', 'a') as fw:
        for sample in dev_sampleset:
            sample_dict = {}
            sample_dict['context'] = sample.context
            sample_dict['response'] = sample.response
            sample_dict['spk_agents'] = sample.spk_agents
            sample_dict['true_adr'] = sample.true_adr
            sample_dict['true_res'] = sample.true_res
            # sample_dict['agent_index_dict'] = sample.agent_index_dict
            sample_json = json.dumps(sample_dict)

            fw.write(sample_json)
            fw.write('\n')
    fw.close()

    print 'dev data finished.'

    test_sampleset = create_samples(argv, dataset[0], vocab_words, test=True)
    test_sampleset = create_samples(argv, dataset[0], vocab_words, test=True)
    with open('test_cand_2.txt', 'a') as fw:
        for sample in test_sampleset:
            sample_dict = {}
            sample_dict['context'] = sample.context
            sample_dict['response'] = sample.response
            sample_dict['spk_agents'] = sample.spk_agents
            sample_dict['true_adr'] = sample.true_adr
            sample_dict['true_res'] = sample.true_res
            # sample_dict['agent_index_dict'] = sample.agent_index_dict
            sample_json = json.dumps(sample_dict)

            fw.write(sample_json)
            fw.write('\n')
    fw.close()

    print 'test data finished.'




