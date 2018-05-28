# encoding:utf-8
import torch
import torch.utils.data as data
import os
torch.manual_seed(1)    # reproducible
import json
import linecache
import numpy as np

class ChatDataset(data.Dataset):
    def __init__(self, file_path, train=True, transform=None):
        super(ChatDataset, self).__init__()
        self.train = train

        if self.train:
            self.train_data = file_path
        else:
            self.test_data, self.test_labels = file_path
        self.transform = transform


    def __getitem__(self, index):

        line = linecache.getline(self.train_data, index+1)
        self.sample = {}
        data = json.loads(line)

        true_response_label = data['true_res']
        if true_response_label == 0:  # response[0] is true
            resp_label = 1
        else:                         # response[1] is true
            resp_label = -1

        # data['response'][true_response_label][-1] = 1
        self.sample['context'] = np.asarray(data['context'])
        self.sample['response'] = np.asarray(data['response'])
        self.sample['resp_label'] = resp_label
        self.sample['true_res'] = data['true_res']
        self.sample['spk_agents'] = np.asarray(data['spk_agents'])
        self.sample['true_adr'] = data['true_adr']

        return self.sample


    def __len__(self):
        count = 0
        for count, line in enumerate(open(self.train_data, 'r')):
            pass
        count += 1
        return count


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        ret = {
                "context": torch.from_numpy(np.array((sample["context"]))),
                "response": torch.from_numpy(np.array(sample["response"])),
                "spk_agents": torch.from_numpy(np.array(sample["spk_agents"])),
                "true_adr": torch.from_numpy(np.array(sample["true_adr"])),
                "true_res": torch.from_numpy(np.array(sample["response"]))
                }

        return ret