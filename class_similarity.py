import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


class Similarity(nn.Module):
    def __init__(self, use_cuda, hidden_size):
        super(Similarity, self).__init__()
        self.use_cuda = use_cuda
        self.fc1 = nn.Linear(2*hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size/2)
        self.fc3 = nn.Linear(hidden_size/2, 1)

    def forward(self, context_hidden, resp_hidden):
        # resp_hidden (n_responses, batch_size, hidden_size)
        # context_hidden (n_directions, batch, hidden_size)

        context_hidden = torch.squeeze(context_hidden)
        score = Variable(torch.zeros(2, len(context_hidden)))
        if self.use_cuda:
            score = score.cuda()

        for index, resp in enumerate(resp_hidden):
            context_resp = torch.cat([context_hidden, resp], 1)  # (batch, 2*hidden)
            fc_out1 = F.tanh(self.fc1(context_resp))
            fc_out2 = F.tanh(self.fc2(fc_out1))
            fc_out3 = F.sigmoid(self.fc3(fc_out2))
            fc_out3 = torch.squeeze(fc_out3)
            score[index] = fc_out3

        return score  # (2, batch_size)


    def cos_similarity(self, context_hidden, resp_hidden):
        batch_size = len(resp_hidden[0])
        n_response = len(resp_hidden)
        self.similarity = Variable(torch.zeros(n_response, batch_size))
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        if self.use_cuda:
            self.similarity, self.cos = self.similarity.cuda(), self.cos.cuda()

        for resp_idx in range(n_response):
            context = torch.squeeze(context_hidden)
            response_i = torch.squeeze(resp_hidden[resp_idx])
            self.similarity[resp_idx] = self.cos(context, response_i)

        log_softmax = nn.LogSoftmax()
        if self.use_cuda:
            log_softmax = log_softmax.cuda()
        out = log_softmax(torch.transpose(self.similarity, 0, 1))  # (batch, 2)

        return out


class Spk_Similarity(nn.Module):
    def __init__(self, use_cuda, hidden_size):
        super(Spk_Similarity, self).__init__()
        self.use_cuda = use_cuda
        self.fc1 = nn.Linear(2 * hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size / 2)
        self.fc3 = nn.Linear(hidden_size / 2, 1)

    def forward(self, spk_emb, resp_true):
        # spk_emb (batch_size, 2, hidden_size)
        # resp_true (batch, hidden_size)
        # n_prev_sent: n_spk
        n_spk = spk_emb.size(1)
        batch_size = spk_emb.size(0)
        spk_emb = torch.transpose(spk_emb, 0, 1)

        score = Variable(torch.zeros(n_spk, batch_size))   # the spk score of every batch
        if self.use_cuda:
            score = score.cuda()
        for index, spk in enumerate(spk_emb):
            spk_resp = torch.cat([spk, resp_true], 1)  # (batch, 2*hidden)
            fc_out1 = F.tanh(self.fc1(spk_resp))
            fc_out2 = F.tanh(self.fc2(fc_out1))
            fc_out3 = self.fc3(fc_out2)
            fc_out3 = torch.squeeze(fc_out3)
            score[index] = fc_out3
        # score = torch.transpose(score, 0, 1)  # (batch_size, n_spk)
        # log_softmax= nn.LogSoftmax()
        # score = log_softmax(score)

        return score


