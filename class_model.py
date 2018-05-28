# encoding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from class_similarity import Similarity, Spk_Similarity

class GroupChatAModel_Adr(nn.Module):
    def __init__(self, use_cuda, embedding, emb_size, hidden_size):
        super(GroupChatAModel_Adr, self).__init__()

        self.use_cuda = use_cuda
        self.rnn_model = AdrResModel(use_cuda, embedding, emb_size, hidden_size)
        self.context_atten = Context_Attention(use_cuda, hidden_size)
        self.spk_atten = Spk_Attention(use_cuda, hidden_size)
        self.similarity = Similarity(use_cuda, hidden_size)
        self.spk_similarity = Spk_Similarity(use_cuda, hidden_size)

    def forward(self, context, response, resp_target, spk_agents, spk_target):
        context_output, context_hidden, resp_output, resp_hidden, spk_emb, spk_emb_mask = self.rnn_model(context, response, spk_agents)

        # context to response attention
        resp_c, alpha_c = self.context_atten(context_output, resp_output, resp_hidden)  # resp_c:(n_response,batch_size,hidden_size)
        score_r = self.similarity(context_hidden, resp_c)  # (batch, 1)

        # spk to response attention
        true_resp = self.get_true_resp(resp_c, resp_target)  # (batch, hidden_size)
        resp_s, alpha_s = self.spk_atten(spk_emb, spk_emb_mask, true_resp)
        spk_emb = self.get_spk_sample(spk_emb, spk_agents, spk_target)
        score_s = self.spk_similarity(spk_emb, resp_s)  # spk_emb (n_prev_sent, batch_size, hidden_size)

        return score_r, score_s

    def get_true_resp(self, resp_c, resp_target):
        true_resp = Variable(torch.zeros(*resp_c[0].size()))
        if self.use_cuda:
            true_resp = true_resp.cuda()

        resp_c = torch.transpose(resp_c, 0, 1)  # (batch_size, n_response, hidden_size)
        for index, true_idx in enumerate(resp_target):
            true_resp[index] = resp_c[index][true_idx]  # (batch_size, hidden_size)

        return true_resp

    def get_spk_sample(self, spk_emb, spk_agents, spk_target):  # a true spk and a random selected false spk
        # spk_emb (batch_size, n_prev_sent, hidden_size)
        # spk_target (batch_size)
        adr_sample = Variable(torch.zeros(spk_emb.size(0), 2, spk_emb.size(2)))  # (batch, 2, hidden_size)
        if self.use_cuda:
            adr_sample = adr_sample.cuda()

        for index, target in enumerate(spk_target):
            target = target.item()
            true_adr = spk_emb[index][target].unsqueeze(0)
            spk_set = {}.fromkeys(spk_agents[index].data.cpu().numpy()).keys()
            np.random.shuffle(np.asarray(spk_set))
            if spk_set[0] != target:
                false_adr = spk_emb[index][spk_set[0]].unsqueeze(0)
            elif len(spk_set)>1:
                false_adr = spk_emb[index][spk_set[1]].unsqueeze(0)
            else:
                false_adr = spk_emb[index][spk_set[0]].unsqueeze(0)
            sample = torch.cat((true_adr, false_adr), dim=0)
            adr_sample[index] = sample

        return adr_sample


class GroupChatModel(nn.Module):
    def __init__(self, use_cuda, embedding, emb_size, hidden_size):
        super(GroupChatModel, self).__init__()

        self.use_cuda = use_cuda
        self.rnn_model = RNNModel(use_cuda, embedding, emb_size, hidden_size)
        self.atten_model = Context_Attention(use_cuda, hidden_size)
        self.similarity = Similarity(use_cuda, hidden_size)

    def forward(self, context, response):
        context_output, context_hidden, resp_output, resp_hidden = self.rnn_model(context, response)
        resp_star, alpha_vec = self.atten_model(context_output, resp_output, resp_hidden)  # resp_atten:(n_response,batch_size,hidden_size)
        score = self.similarity(context_hidden, resp_star)  # (batch, 1)
        return score


class AdrResModel(nn.Module):
    def __init__(self, use_cuda, embedding, emb_size, hidden_size):
        super(AdrResModel, self).__init__()

        self.use_cuda = use_cuda
        self.embedding = embedding
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.utter_rnn = BasicRNN(use_cuda, embedding, emb_size, hidden_size)
        self.context_rnn = BasicRNN(use_cuda, None, 2*hidden_size, hidden_size)
        self.resp_rnn = BasicRNN(use_cuda, embedding, emb_size, hidden_size)
        self.spk_gru = nn.GRU(hidden_size, hidden_size)
        # if self.use_cuda:
        #     self.flatten_parameters()


    def forward(self, context, response, spk_agents):
        # context: (batch, n_prev_sents, max_n_words)
        # response: (batch, n_responses, max_n_words)
        # spk_agents: (batch, n_prev_sents)

        context_output, context_hidden = self.context_encoder(context, spk_agents)
        # context_output: (n_prev_sent, batch, hidden_size)
        # context_hidden: (num_layers * num_directions, batch, hidden_size)

        spk_emb, spk_emb_mask = self.spk_encoder(context_output, spk_agents)  # (batch_size, n_prev_sents, hidden_size)

        resp_output, resp_hidden = self.response_encoder(response)

        return context_output, context_hidden, resp_output, resp_hidden, spk_emb, spk_emb_mask


    def context_encoder(self, context, spk_agents):
        n_prev_sents = len(context[1])
        batch_size = len(context)

        # utter_rnn
        utter_hidden = Variable(torch.zeros(n_prev_sents, batch_size, self.hidden_size))
        if self.use_cuda:
            utter_hidden = utter_hidden.cuda()

        utter_hidden_init = self.utter_rnn.initHidden(batch_size)
        for utter_idx in range(n_prev_sents):
            utter_input = context[:, utter_idx, :]  # (batch, max_n_words)
            utter_input = self.utter_rnn.embedding(utter_input)
            utter_input = torch.transpose(utter_input, 0, 1)  # (max_n_words, batch, emb_size)
            output, hidden = self.utter_rnn(utter_input, utter_hidden_init)
            # output: (max_n_words, batch, hidden_size * num_directions)
            # hidden: (num_layers * num_directions, batch, hidden_size)
            utter_hidden[utter_idx] = torch.squeeze(hidden)

        # agent embedding
        spk_agents_input = self.get_agent_emb(spk_agents, self.emb_size)  # (batch, n_prev_sents, emb_size)
        spk_agents_input = torch.transpose(spk_agents_input, 0, 1)  # (n_prev_sents, batch, emb_size)

        # context_rnn
        context_input = torch.cat([utter_hidden, spk_agents_input], 2)  # (n_prev_sents, batch, 2 * emb_size)
        context_hidden_init = self.context_rnn.initHidden(batch_size)
        context_output, context_hidden = self.context_rnn(context_input, context_hidden_init)

        return context_output, context_hidden


    def response_encoder(self, response):
        batch_size = response.size(0)
        n_responses = response.size(1)
        n_words = response.size(2)

        # resp_rnn
        resp_output = Variable(torch.zeros(n_responses, n_words, batch_size, self.hidden_size))
        resp_hidden = Variable(torch.zeros(n_responses, batch_size, self.hidden_size))
        if self.use_cuda:
            resp_output, resp_hidden = resp_output.cuda(), resp_hidden.cuda()

        resp_hidden_init = self.resp_rnn.initHidden(batch_size)
        for resp_idx in range(n_responses):
            resp_input = response[:, resp_idx, :]  # (batch, max_n_words)
            resp_input = self.resp_rnn.embedding(resp_input)
            resp_input = torch.transpose(resp_input, 0, 1)  # (max_n_words, batch, embedding_size)
            output, hidden = self.resp_rnn(resp_input, resp_hidden_init)

            resp_output[resp_idx] = torch.squeeze(output)
            resp_hidden[resp_idx] = torch.squeeze(hidden)

        return resp_output, resp_hidden


    def spk_encoder(self, context_output, spk_agents):
        # context_output (n_prev_sent, batch, hidden_size)
        # spk_agents (batch, n_prev_sents)  (n_prev_sents,  batch) to one-hot

        n_prev_sents = context_output.size(0)
        batch_size = context_output.size(1)
        hidden_size = context_output.size(2)

        # spk_emb: spk_len=n_prev_sents+1
        spk_emb = Variable(torch.zeros(batch_size, n_prev_sents+1, hidden_size))
        if self.use_cuda:
            spk_emb = spk_emb.cuda()
        # spk_gru = nn.GRU(hidden_size, hidden_size)
        for batch in range(context_output.size(1)):  # batch_size
            spk_batch = spk_agents[batch].data.cpu().numpy()  # [1, n_prev_sents]
            spk_set = {}.fromkeys(spk_batch).keys()

            # one spk emb
            one_spk_emb = Variable(torch.zeros(n_prev_sents+1, hidden_size))
            if self.use_cuda:
                one_spk_emb = one_spk_emb.cuda()
            for spk in spk_set:
                spk_index = [i for i in range(n_prev_sents) if spk_batch[i] == spk]
                input = context_output[:, batch, :][spk_index,].unsqueeze(1)   # (n_spk, hidden_size)
                _, hidden = self.spk_gru(input)  # （1，batcb=1, hidden_size）(num_layers * num_directions, batch, hidden_size)
                one_spk_emb[spk] = hidden.squeeze(1)
            spk_emb[batch] = one_spk_emb
        spk_emb_mask = self.get_mask(spk_emb)

        return spk_emb, spk_emb_mask  # no repeat


    def spk_encoder1(self, context_output, spk_agents):
        # context_output (n_prev_sent, batch, hidden_size)
        # spk_agents (batch, n_prev_sents)  (n_prev_sents,  batch) to one-hot

        n_prev_sents = context_output.size(0)
        batch_size = context_output.size(1)
        hidden_size = context_output.size(2)

        # spk_emb
        spk_emb = Variable(torch.zeros(batch_size, n_prev_sents, hidden_size))
        if self.use_cuda:
            spk_emb = spk_emb.cuda()
        spk_gru = nn.GRU(hidden_size, hidden_size)
        for batch in range(context_output.size(1)):  # batch_size
            spk_batch = spk_agents[batch].data.cpu().numpy()  # [1, n_prev_sents]
            spk_set = {}.fromkeys(spk_batch).keys()

            # one spk emb
            spk_emb_temp = Variable(torch.zeros(n_prev_sents, hidden_size))
            if self.use_cuda:
                spk_emb_temp = spk_emb_temp.cuda()
            for spk in spk_set:
                spk_index = [i for i in range(n_prev_sents) if spk_batch[i] == spk]
                input = context_output[:, batch, :][spk_index].unsqueeze(1)   # (n_spk, hidden_size)
                _, hidden = spk_gru(input)  # （1，batcb=1, hidden_size）(num_layers * num_directions, batch, hidden_size)
                for i in spk_index:
                    spk_emb_temp[i] = hidden.squeeze(1)
            spk_emb[batch] = spk_emb_temp

        return spk_emb  # no repeat


    def get_agent_emb(self, spk_agents, emb_size):
        # spk_agents: (batch, n_prev_sents)

        n_prev_sents = spk_agents.size(1)

        np.random.seed(1)
        agent_emb = Variable(torch.zeros(spk_agents.size(0), spk_agents.size(1), emb_size))
        if self.use_cuda:
            agent_emb = agent_emb.cuda()

        for batch in range(spk_agents.size(0)):
            agents_set_i = spk_agents[batch]  # (1, n_prev_sents)
            emb_table = np.random.uniform(0, 1, (n_prev_sents+1, emb_size))  # n_agent = n_prev_sent+1

            # agents_set_i: 1,0,2,0,1
            for i, agent in enumerate(agents_set_i):  # [4,4,3,2,1]
                agent_emb[batch][i] = torch.from_numpy(emb_table[agent])


            # embedding = nn.Embedding(n_agents, emb_size, padding_idx=0)
            # embedding.weight.data.copy_(torch.from_numpy(emb_table))
            # embedding.weight.requires_grad = True
            # if self.use_cuda:
            #     embedding = embedding.cuda()

            # agent_emb[batch] = embedding(torch.unsqueeze(agents_set_i, 0)).data  # (1, n_prev_sents, emb_size)

        return agent_emb


    def get_mask(self, seq):
        # seq : (batch_size, n_prev_sents, hidden_size)
        batch_size = seq.size(0)
        n_spk = seq.size(1)

        mask = Variable(torch.zeros(batch_size, n_spk))
        tensor_z = Variable(torch.zeros(seq.size(2)))
        if self.use_cuda:
            mask, tensor_z = mask.cuda(), tensor_z.cuda()

        for batch, seq_batch in enumerate(seq):
            for i in range(n_spk):
                if i == 0:
                    mask[batch, i] = 0
                elif not torch.equal(seq_batch[i], tensor_z):
                    mask[batch, i] = 1

        return mask


class Context_Attention(nn.Module):
    def __init__(self, use_cuda, hidden_size):
        super(Context_Attention, self).__init__()
        self.use_cuda = use_cuda
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_p = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_x = nn.Linear(hidden_size, hidden_size, bias=False)
        if self.use_cuda:
            self.linear, self.linear_p, self.linear_x = self.linear.cuda(), self.linear_p.cuda(), self.linear_x.cuda()

        # attention parameters
        if self.use_cuda:
            self.W_c = nn.Parameter(torch.randn(hidden_size, hidden_size).cuda())
            self.W_t = nn.Parameter(torch.randn(hidden_size, hidden_size).cuda())
            self.W_a = nn.Parameter(torch.randn(hidden_size, hidden_size).cuda())
            self.W_alpha = nn.Parameter(torch.randn(hidden_size, 1).cuda())
        else:
            self.W_c = nn.Parameter(torch.randn(hidden_size, hidden_size))
            self.W_t = nn.Parameter(torch.randn(hidden_size, hidden_size))
            self.W_a = nn.Parameter(torch.randn(hidden_size, hidden_size))
            self.W_alpha = nn.Parameter(torch.randn(hidden_size, 1))

        self.register_parameter('W_c', self.W_c)
        self.register_parameter('W_t', self.W_t)
        self.register_parameter('W_a', self.W_a)
        self.register_parameter('W_alpha', self.W_alpha)


    def forward(self, context_output, resp_output, resp_hidden):
        alpha_vec = []
        resp_c = Variable(torch.zeros(*resp_hidden.size()))  # (n_response, batch, hidden_size)
        if self.use_cuda:
            resp_c = resp_c.cuda()

        for index, resp in enumerate(resp_output):
            resp_atten_i, alpha_vec_i = self._resp_forward(context_output, resp)  # (batch, hidden_size)
            res_hidden_i = resp_hidden[index]
            resp_c_i = F.tanh(self.linear_p(resp_atten_i) + self.linear_x(res_hidden_i))
            resp_c[index] = resp_c_i
            alpha_vec.append(alpha_vec_i)

        return resp_c, alpha_vec


    def _resp_forward(self, context_output, resp_output):
        # resp_output: (n_words, batch_size, self.hidden_size)  one response
        n_words = resp_output.size(0)
        batch_size = resp_output.size(1)
        n_prev_sents = context_output.size(0)

        alpha_c = Variable(torch.zeros(n_words, batch_size, n_prev_sents))
        resp_atten = Variable(torch.zeros(batch_size, self.hidden_size))  # init response attention
        if self.use_cuda:
            alpha_c, resp_atten = alpha_c.cuda(), resp_atten.cuda()

        for index, resp_t in enumerate(resp_output):
            atten_t, alpha = self._atten_forward(context_output, resp_t, resp_atten)  # (batch, hidden_size)
            alpha_c[index] = alpha
            resp_atten = atten_t + F.tanh(self.linear(resp_atten))

        return resp_atten, alpha_c

    def _atten_forward(self, context_output, resp_t, resp_atten=None):
        # context_output: (n_prev_sents, batch, hidden_size)
        # resp_t: (batch, hidden_size) the t-th word in response
        context_output = context_output.transpose(1, 0)  # (batch, n_prev_sents, hidden_size)
        batch_size = context_output.size(0)
        n_prev_sents = context_output.size(1)

        W_context = torch.bmm(context_output, self.W_c.unsqueeze(0).expand(batch_size, *self.W_c.size())) # (batch, n_prev_sents, hidden_size)
        W_resp_t = torch.mm(resp_t, self.W_t)  # (batch, hidden_size)
        if resp_atten is not None:
            W_atten = torch.mm(resp_atten, self.W_a)  # (batch, hidden_size)
            W_resp_t += W_atten
        M = torch.tanh(W_context + W_resp_t.unsqueeze(1).expand(W_resp_t.size(0),n_prev_sents,W_resp_t.size(1))) # (batch, n_prev_sents, hidden_size)
        alpha = torch.bmm(M, self.W_alpha.unsqueeze(0).expand(batch_size, *self.W_alpha.size())).squeeze(-1)  # (batch, n_prev_sents)
        # alpha = alpha + (-1000.0 * (1. - mask_Y))
        alpha = F.softmax(alpha, 1)  # (batch, n_prev_sents)
        atten_t = torch.bmm(alpha.unsqueeze(1), context_output).squeeze(1)  # (batch, hidden_size)

        return atten_t, alpha


class Spk_Attention(nn.Module):
    def __init__(self, use_cuda, hidden_size):
        super(Spk_Attention, self).__init__()
        self.use_cuda = use_cuda
        self.hidden_size = hidden_size
        self.linear_p = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_x = nn.Linear(hidden_size, hidden_size, bias=False)
        if self.use_cuda:
            self.linear_p, self.linear_x = self.linear_p.cuda(), self.linear_x.cuda()

        # attention parameters
        if self.use_cuda:
            self.W_s = nn.Parameter(torch.randn(hidden_size, hidden_size).cuda())
            self.W_r = nn.Parameter(torch.randn(hidden_size, hidden_size).cuda())
            self.w = nn.Parameter(torch.randn(hidden_size, 1).cuda())
        else:
            self.W_s = nn.Parameter(torch.randn(hidden_size, hidden_size))
            self.W_r = nn.Parameter(torch.randn(hidden_size, hidden_size))
            self.w = nn.Parameter(torch.randn(hidden_size, 1))

        self.register_parameter('W_s', self.W_s)
        self.register_parameter('W_r', self.W_r)
        self.register_parameter('w', self.w)

    def forward(self, spk_emb, spk_emb_mask, true_resp):
        # spk_emb (batch_size, n_prev_sent, hidden_size)
        # spk_emb_mask (batch_size, n_prev_sent)
        # true_resp (batch_size, hidden_size)

        batch_size = spk_emb.size(0)
        n_spk = spk_emb.size(1)
        hidden_size = spk_emb.size(2)

        W_spk = torch.bmm(spk_emb, self.W_s.unsqueeze(0).expand(batch_size, *self.W_s.size()))
        W_resp = torch.mm(true_resp, self.W_r)  # (batch, hidden_size)
        W_resp = W_resp.unsqueeze(1).expand(batch_size, n_spk, hidden_size)  # (batch, n_prev_sents, hidden_size)
        spk_emb_mask = spk_emb_mask.unsqueeze(2).expand(batch_size, n_spk, hidden_size)

        M = torch.tanh(W_spk + W_resp)
        M = M * spk_emb_mask

        alpha_s = torch.bmm(M, self.w.unsqueeze(0).expand(batch_size, *self.w.size())).squeeze(-1)
        alpha_s = F.softmax(alpha_s, 1)  # (batch, n_prev_sents)
        resp_s = torch.bmm(alpha_s.unsqueeze(1), spk_emb).squeeze(1)  # (batch, hidden_size)
        resp_s = F.tanh(self.linear_p(resp_s) + self.linear_x(true_resp))

        return resp_s, alpha_s


    def _spk_forward(self, spk_emb, true_resp):
        # spk_emb: (batch_size, n_prev_sents, hidden_size)
        # resp_hidden: (1, hidden_size)  one response

        batch_size = spk_emb.size(0)
        n_prev_sents = spk_emb.size(1)

        W_spk = torch.bmm(spk_emb, self.W_s.unsqueeze(0).expand(batch_size, *self.W_s.size()))
        W_resp = torch.mm(true_resp, self.W_r)  # (batch, hidden_size)
        W_resp = W_resp.unsqueeze(1).expand(W_resp.size(0), n_prev_sents, W_resp.size(1))  # (batch, n_prev_sents, hidden_size)
        M = torch.tanh(W_spk + W_resp)
        alpha = torch.bmm(M, self.w.unsqueeze(0).expand(batch_size, *self.w.size())).squeeze(-1)
        alpha = F.softmax(alpha)  # (batch, n_prev_sents)
        resp_s = torch.bmm(alpha.unsqueeze(1), spk_emb).squeeze(1)  # (batch, hidden_size)

        return resp_s, alpha


class RNNModel(nn.Module):
    def __init__(self, use_cuda, embedding, emb_size, hidden_size):
        super(RNNModel, self).__init__()

        self.use_cuda = use_cuda
        self.embedding = embedding
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.utter_rnn = BasicRNN(use_cuda, embedding, emb_size, hidden_size)
        self.context_rnn = BasicRNN(use_cuda, None, hidden_size, hidden_size)
        self.resp_rnn = BasicRNN(use_cuda, embedding, emb_size, hidden_size)

    def forward(self, context, response):
        # context: (batch, n_prev_sents, max_n_words)
        # response: (batch, n_responses, max_n_words)
        # response_label: (batch, n_responses)

        n_prev_sents = len(context[1])
        n_words = context.size(2)
        n_responses = len(response[1])
        batch_size = len(context)

        # utter_rnn
        utter_hidden = Variable(torch.zeros(n_prev_sents, batch_size, self.hidden_size))
        if self.use_cuda:
            utter_hidden = utter_hidden.cuda()

        utter_hidden_init = self.utter_rnn.initHidden(batch_size)
        for utter_idx in range(n_prev_sents):
            utter_input = context[:, utter_idx, :]  # (batch, max_n_words)
            utter_input = self.utter_rnn.embedding(utter_input)
            utter_input = torch.transpose(utter_input, 0, 1)  # (max_n_words, batch, emb_size)
            output, hidden = self.utter_rnn(utter_input, utter_hidden_init)
            # output: (max_n_words, batch, hidden_size * num_directions)
            # hidden: (num_layers * num_directions, batch, hidden_size)
            utter_hidden[utter_idx] = torch.squeeze(hidden)

        # context_rnn
        context_hidden_init = self.context_rnn.initHidden(batch_size)
        context_output, context_hidden = self.context_rnn(utter_hidden, context_hidden_init)
        # context_output: (n_prev_sent, batch, hidden_size)
        # context_hidden: (num_layers * num_directions, batch, hidden_size)

        # resp_rnn
        resp_output = Variable(torch.zeros(n_responses, n_words, batch_size, self.hidden_size))
        resp_hidden = Variable(torch.zeros(n_responses, batch_size, self.hidden_size))
        if self.use_cuda:
            resp_output, resp_hidden = resp_output.cuda(), resp_hidden.cuda()

        resp_hidden_init = self.resp_rnn.initHidden(batch_size)
        for resp_idx in range(n_responses):
            resp_input = response[:, resp_idx, :]  # (batch, max_n_words)
            resp_input = self.resp_rnn.embedding(resp_input)
            resp_input = torch.transpose(resp_input, 0, 1)  # (max_n_words, batch, embedding_size)
            output, hidden = self.resp_rnn(resp_input, resp_hidden_init)

            resp_output[resp_idx] = torch.squeeze(output)
            resp_hidden[resp_idx] = torch.squeeze(hidden)

        return context_output, context_hidden, resp_output, resp_hidden


class BasicRNN(nn.Module):
    def __init__(self, use_cuda, embeddings, input_size, hidden_size):
        super(BasicRNN, self).__init__()
        self.use_cuda = use_cuda
        self.hidden_size = hidden_size

        if embeddings is not None:
            pretrained_weight = np.array(embeddings)
            self.embedding = nn.Embedding(pretrained_weight.shape[0], pretrained_weight.shape[1], padding_idx=0)
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
            self.embedding.weight.requires_grad = True
            if self.use_cuda:
                self.embedding = self.embedding.cuda()


        self.gru = nn.GRU(input_size, hidden_size)
        if self.use_cuda:
            self.gru = self.gru.cuda()


    def forward(self, input, h0):
        # input shape (batch, max_n_words)
        # h0: (1, 20, 32)

        # input = input.long()  # (batch, max_n_words)
        # input = torch.transpose(self.embedding(input),0,1)  # (batch, max_n_words, embedding_size) (4,20,10)
        # output: (seq_len, batch, hidden_size * num_directions)
        # hidden: (num_layers * num_directions, batch, hidden_size)
        output, hidden = self.gru(input, h0)
        return output, hidden


    def initHidden(self, batch_size):  # batch_size动态变化
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if self.use_cuda:
            return result.cuda()
        else:
            return result
