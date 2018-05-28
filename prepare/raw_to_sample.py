# encoding:utf-8
from load import load_dataset, get_samples
import json


raw_train_data = "../data/input/train-data-test"
raw_dev_data = "../data/input/dev-data-test"
raw_test_data = "../data/input/test-data-test"
data_size = 100000
n_prev_sents = 5   # 上下文长度
max_n_words = 20   # 句子长度


if __name__ == "__main__":
    # load dataset
    train_data = load_dataset(raw_train_data,data_size)
    dev_data = load_dataset(raw_train_data, data_size)
    test_data = load_dataset(raw_train_data, data_size)

    # create_samples
    train_samples = get_samples(threads=train_data, n_prev_sents=n_prev_sents, max_n_words=max_n_words, pad=False)
    with open('train_cand_2.txt', 'a') as fw:
        for sample in train_samples:
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

