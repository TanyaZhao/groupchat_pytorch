import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

use_cuda = torch.cuda.is_available()

class RNN(nn.Module):
    def __init__(self, embeddings, input_size, hidden_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        if embeddings is not None:
            pretrained_weight = np.array(embeddings)
            self.embedding = nn.Embedding(pretrained_weight.shape[0], pretrained_weight.shape[1], padding_idx=0).cuda()
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))

        self.gru = nn.GRU(input_size, hidden_size).cuda()


    def forward(self, input, h0):
        # input shape (batch, max_n_words)
        # h0: (1, 20, 32)

        # input = input.long()  # (batch, max_n_words)
        # input = torch.transpose(self.embedding(input),0,1)  # (batch, max_n_words, embedding_size) (4,20,10)
        # output: (seq_len, batch, hidden_size * num_directions)
        # hidden: (num_layers * num_directions, batch, hidden_size)
        output, hidden = self.gru(input, h0)
        return output, hidden


    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        # return result
        if torch.cuda.is_available():
            return result.cuda()
        else:
            return result