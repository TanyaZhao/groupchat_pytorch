import gzip
from sample import Sample



def create_samples(dataset, n_prev_sents, max_n_words, test = False):
    ###########################
    # Task setting parameters #
    ###########################
    cands = dataset[0][0][3:-1]
    n_cands = len(cands)

    ##################
    # Create samples #
    ##################
    # samples: 1D: n_samples; 2D: Sample
    samples = get_samples(threads=dataset, n_prev_sents=n_prev_sents, max_n_words=max_n_words)

    return samples


def get_samples(threads, n_prev_sents, max_n_words=1000, pad=True, test=False):

    if threads is None:
        return None

    samples = []
    max_n_agents = n_prev_sents + 1

    for thread in threads:
        samples += get_one_thread_samples(thread, max_n_words, max_n_agents, n_prev_sents, pad, test)

    return samples


def get_one_thread_samples(thread, max_n_words, max_n_agents, n_prev_sents, pad=True, test=False):
    samples = []
    sents = []
    agents_in_ctx = set([])

    for i, sent in enumerate(thread):
        time = sent[0]
        spk_id = sent[1]
        adr_id = sent[2]
        label = sent[-1]

        context = get_context(i, sents, n_prev_sents, label, test)
        responses = limit_sent_length(sent[3:-1], max_n_words)

        original_sent = get_original_sent(responses, label)
        sents.append((time, spk_id, adr_id, original_sent))

        agents_in_ctx.add(spk_id)

        ################################
        # Judge if it is sample or not #
        ################################
        if is_sample(context, spk_id, adr_id, agents_in_ctx):
            sample = Sample(context=context, spk_id=spk_id, adr_id=adr_id, responses=responses, label=label,
                            n_agents_in_ctx=len(agents_in_ctx), max_n_agents=max_n_agents, max_n_words=max_n_words,
                            pad=pad, test=test)
            if test:
                samples.append(sample)
            else:
                # The num of the agents in the training samples is n_agents > 1
                # -1 means that the addressee does not appear in the limited context
                if sample.true_adr > -1:
                    samples.append(sample)

    return samples


def get_original_sent(responses, label):
    if label > -1:
        return responses[label]
    return responses[0]


def limit_sent_length(sents, max_n_words):
    return [sent[:max_n_words] for sent in sents]


def is_sample(context, spk_id, adr_id, agents_in_ctx):
    if context is None:
        return False
    if spk_id == adr_id:
        return False
    if adr_id not in agents_in_ctx:
        return False
    return True


def get_context(i, sents, n_prev_sents, label, test=False):
    # context: 1D: n_prev_sent, 2D: (time, speaker_id, addressee_id, tokens, label)
    context = None
    if label > -1:
        if len(sents) >= n_prev_sents:
            context = sents[i - n_prev_sents:i]
        elif test:
            context = sents[:i]
    return context




def load_dataset(fn, data_size=1000000):
    """
    :param fn: file name
    :param vocab: vocab set
    :param data_size: how many threads are used
    :return: threads: 1D: n_threads, 2D: n_utterances, 3D: elem=(time, speaker_id, addressee_id, cand_res1, ... , label)
    """
    if fn is None:
        return None

    threads = []
    thread = []
    file_open = gzip.open if fn.endswith(".gz") else open

    with file_open(fn) as gf:
        # line: (time, speaker_id, addressee_id, cand_res1, cand_res2, ... , label)
        for line in gf:
            line = line.rstrip().split("\t")

            if len(line) < 6:
                threads.append(thread)
                thread = []

                if len(threads) >= data_size:
                    break
            else:
                for i, sent in enumerate(line[3:-1]):
                    words = []
                    for w in sent.split():
                        w = w.lower()
                        words.append(w)
                    line[3 + i] = words

                ##################
                # Label          #
                # -1: Not sample #
                # 0-: Sample     #
                ##################
                line[-1] = -1 if line[-1] == '-' else int(line[-1])
                thread.append(line)

    return threads
