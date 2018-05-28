# coding:utf-8
import sys
import numpy as np
import random
from sample import Sample


def get_samples(dataset, n_prev_sents, max_n_words=1000, pad=True, test=False):
    """
    :param threads: 1D: n_docs, 2D: n_utters, 3D: (time, speaker_id, addressee_id, response, ..., label)
    :param n_prev_sents: how many previous sentences are used
    :param max_n_words: how many words in a utterance are used
    :param pad: whether do the zero padding or not
    :param test: whether the dev/test set or not
    :return: samples: 1D: n_samples; elem=Sample()
    """
    if dataset is None:
        return None

    print('\n\tn_docs: {:>5}\n'.format(len(dataset)))

    samples = []
    max_n_agents = n_prev_sents + 1

    count = 0
    for doc in dataset:
        samples += get_one_doc_samples(doc, max_n_words, max_n_agents, n_prev_sents, pad, test)
        count += 1
        if test and count == len(dataset)/2:
            return samples
    return samples


def get_one_doc_samples(doc, max_n_words, max_n_agents, n_prev_sents, pad=True, test=False):
    samples = []
    sents = []
    agents_in_ctx = set([])

    for i, sent in enumerate(doc):  # for every utterance in doc
        time = sent[0]
        spk_id = sent[1]
        adr_id = sent[2]
        label = sent[-1]

        context = get_context(i, sents, n_prev_sents, label, test)
        responses = limit_sent_length(sent[3:-1], max_n_words)

        original_sent = get_original_sent(responses, label)
        sents.append((time, spk_id, adr_id, original_sent))

        agents_in_ctx.add(spk_id)
        # if context is not None:
        #     agents_in_ctx = [c[1] for c in context]
        #     agents_in_ctx = list(set(agents_in_ctx))

        ################################
        # Judge if it is sample or not #
        ################################
        if is_sample(context, responses, spk_id, adr_id, agents_in_ctx):
            sample = Sample(context=context, spk_id=spk_id, adr_id=adr_id, responses=responses, label=label,
                            n_agents_in_ctx=len(agents_in_ctx), max_n_agents=max_n_agents, max_n_words=max_n_words, test=test)
            if test:
                samples.append(sample)
            else:
                # The num of the agents in the training samples is n_agents > 1
                # -1 means that the addressee does not appear in the limited context
                if sample.true_adr > -1:
                    samples.append(sample)

    return samples


def is_sample(context, responses, spk_id, adr_id, agents_in_ctx):
    if context is None:
        return False
    if responses is None:
        return False
    if spk_id == adr_id:
        return False
    if adr_id not in agents_in_ctx:
        return False
    return True


def get_context(i, sents, n_prev_sents, label, test=False):
    # context: 1D: n_prev_sent, 2D: (time, speaker_id, addressee_id, tokens, label)
    context = None
    if label > -1: # label=0/1
        if len(sents) >= n_prev_sents:
            context = sents[i - n_prev_sents:i]
        elif test:
            context = sents[:i]
    return context


def get_original_sent(responses, label):
    if label > -1:
        return responses[label]
    return responses[0]


def limit_sent_length(sents, max_n_words):
    return [sent[:max_n_words] for sent in sents]


def get_max_n_words(context, response):
    return np.max([len(r) for r in response] + [len(c[-1]) for c in context])


def batch_indexing(vec, batch_size, n_labels):
    indexed_vec = []
    for v in vec:
        if v > -1:
            v += batch_size * n_labels
        else:
            v = -1
        indexed_vec.append(v)
    return indexed_vec


def shuffle_batch(samples):
    """
    :param samples: 1D: n_samples; elem=Sample
    :return: 1D: n_samples; elem=Sample
    """

    shuffled_samples = []
    batch = []
    prev_n_agents = samples[0].n_agents_in_lctx
    prev_n_prev_sents = len(samples[0].context)

    for sample in samples:
        n_agents = sample.n_agents_in_lctx
        n_prev_sents = len(sample.context)

        if n_agents != prev_n_agents or n_prev_sents != prev_n_prev_sents:
            np.random.shuffle(batch)
            shuffled_samples += batch

            batch = []
            prev_n_agents = n_agents
            prev_n_prev_sents = n_prev_sents

        batch.append(sample)

    if batch:
        np.random.shuffle(batch)
        shuffled_samples += batch

    return shuffled_samples


def tf_samples(samples, batch_size, test=False):
    """
    :param samples: 1D: n_samples; elem=Sample
    :param batch_size: int
    :return shared_sampleset: 1D: 8, 2D: n_baches, 3D: batch
    """
    samples.sort(key=lambda sample: sample.n_agents_in_lctx)
    samples = shuffle_batch(samples)

    #sampleset = [[] for i in xrange(6)]
    sampleset = [[] for i in xrange(7)]
    evalset = []
    #batch = [[] for i in xrange(5)]
    batch = [[] for i in xrange(6)]
    binned_n_agents = []
    labels_a = []
    labels_r = []

    prev_n_agents = samples[0].n_agents_in_lctx

    for sample in samples:
        n_agents = sample.n_agents_in_lctx

        if len(batch[0]) == batch_size:
            sampleset[0].append(batch[0])
            sampleset[1].append(batch[1])
            sampleset[2].append(batch[2])
            sampleset[3].append(batch[3])
            sampleset[4].append(batch[4])
            sampleset[5].append(batch[5])
            sampleset[6].append(prev_n_agents)

            evalset.append((binned_n_agents,
                            labels_a,
                            labels_r))

            #batch = [[] for i in xrange(5)]
            batch = [[] for i in xrange(6)]
            binned_n_agents = []
            labels_a = []
            labels_r = []

        if n_agents != prev_n_agents:
            #batch = [[] for i in xrange(5)]
            batch = [[] for i in xrange(6)]
            prev_n_agents = n_agents
            binned_n_agents = []
            labels_a = []
            labels_r = []

        ##################
        # Create a batch #
        ##################
        batch[0].append(sample.context)
        batch[1].append(sample.response)
        batch[2].append(sample.spk_agent_one_hot_vec)
        batch[3].append(sample.adr_agent_one_hot_vec)
        # batch[4].append(sample.response_label)
        batch[4].append(sample.true_res)
        batch[5].append(batch_indexing(sample.adr_label_vec, batch_size=len(batch[5]), n_labels=n_agents - 1))

        binned_n_agents.append(sample.binned_n_agents_in_ctx)
        labels_a.append(sample.true_adr)
        labels_r.append(sample.true_res)

    n_batches = len(sampleset[-1])

    return sampleset, evalset, n_batches



def select_random_spk(true_addr, agent_index_dict):
    # true_addr_id = agent_index_dict.keys()[agent_index_dict.values().index(true_addr)]
    # temp = agent_index_dict
    # del temp[true_addr_id]
    agent_lst = agent_index_dict.values()
    index = agent_lst.index(true_addr)
    del agent_lst[index]
    random.shuffle(agent_lst)
    return agent_lst[0]