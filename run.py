# encoding:utf-8
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from chat_dataset import ChatDataset
import torch.nn.functional as F
import torch.utils.data as data
from prepare.prepare_data import get_embeddings
import os
torch.manual_seed(1)    # reproducible

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

TRAIN_DATA = 'train_cand_2_test_old.txt'
DEV_DATA = 'dev_cand_2_test_old.txt'
TEST_DATA = 'test_data_2_test.txt'
INIT_EMB = None
BATCH_SIZE = 32
HIDDEN_SIZE = 100
EMBEDDING_SIZE = 100
EMBEDDING = None
EPOCHES = 19
LEARNING_RATE = 0.01

use_cuda = torch.cuda.is_available()


def train(utter_input, utter_rnn, utter_rnn_optimizer,
          context_rnn, context_rnn_optimizer,
          resp_input, resp_rnn, resp_rnn_optimizer,
          resp_label, criterion, train=True):

    # utter_input (batch, n_prev_sents, max_n_words)
    # resp_input (batch, n_responses, max_n_words)
    # resp_label (batch, n_responses)

    batch_size = len(utter_input)
    utter_hidden_init = utter_rnn.initHidden(batch_size)  # change to spker （layer, batch_size, hidden_size）
    utter_rnn_optimizer.zero_grad()
    context_hidden_init = context_rnn.initHidden(batch_size)  # change to spker （layer, batch_size, hidden_size）
    context_rnn_optimizer.zero_grad()
    resp_hidden_init = resp_rnn.initHidden(batch_size)  # change to spker （layer, batch_size, hidden_size）
    resp_rnn_optimizer.zero_grad()

    loss = 0

    n_prev_sents = len(utter_input[1])
    n_responses = len(resp_input[1])
    utter_hidden = Variable(torch.zeros(n_prev_sents, batch_size, HIDDEN_SIZE)).cuda()
    for utter_idx in range(n_prev_sents):
        input_variable = utter_input[:,utter_idx,:]  # (batch, 1, max_n_words)
        # input_variable = torch.squeeze(input_variable).long()  # (batch, max_n_words)
        input_variable = Variable(input_variable)
        # print input_variable
        # if use_cuda:
        #     print '*' * 20
        input_variable_emb = utter_rnn.embedding(input_variable.cuda())
        input_variable = torch.transpose(input_variable_emb, 0, 1)
        output, hidden = utter_rnn(input_variable, utter_hidden_init)
        # output: (max_n_words, batch, hidden_size * num_directions)
        # hidden: (num_layers * num_directions, batch, hidden_size)

        # utter_output.append(output)
        utter_hidden[utter_idx] = torch.squeeze(hidden)

    context_output, context_hidden = context_rnn(utter_hidden, context_hidden_init)
    # context_output: (n_prev_sent, batch, hidden_size)
    # context_hidden: (num_layers * num_directions, batch, hidden_size)

    similarity = Variable(torch.zeros(n_responses, batch_size)).cuda()
    cos = nn.CosineSimilarity(dim=1, eps=1e-6).cuda()
    for resp_idx in range(n_responses):
        input_variable = resp_input[:, resp_idx, :] # (batch, 1, max_n_words)
        # input_variable = torch.squeeze(input_variable).long()  # (batch, max_n_words)
        input_variable = Variable(input_variable).cuda()
        input_variable = torch.transpose(resp_rnn.embedding(input_variable), 0, 1)  # (max_n_words, batch, embedding_size)
        resp_output, resp_hidden = resp_rnn(input_variable, resp_hidden_init)

        similarity[resp_idx] = cos(torch.squeeze(context_hidden), torch.squeeze(resp_hidden))  # (batch)

    m = nn.LogSoftmax().cuda()
    softmax_cos = m(torch.transpose(similarity, 0, 1))  # (batch, 2)
    resp_label = Variable(resp_label).cuda()

    resp_predict = torch.max(softmax_cos, 1)[1]

    accuracy = (sum(resp_label == resp_predict).float().data.cpu().numpy() / len(resp_predict))[0]

    loss = criterion(softmax_cos, resp_label)  # NLLLoss

    print '*' * 50
    print "utter_rnn.parameters:"
    for para in utter_rnn.parameters():
        print para
        break
    # print "context_rnn.parameters:"
    # for para in context_rnn.parameters():
    #     print para
    #     break
    # print "resp_rnn.parameters:"
    # for para in utter_rnn.parameters():
    #     print para
    #     break


    if train:
        loss.backward()
        utter_rnn_optimizer.step()
        context_rnn_optimizer.step()
        resp_rnn_optimizer.step()

    print "utter_rnn.parameters:"
    for para in utter_rnn.parameters():
        print para
        break
    # print "context_rnn.parameters:"
    # for para in context_rnn.parameters():
    #     print para
    #     break
    # print "resp_rnn.parameters:"
    # for para in utter_rnn.parameters():
    #     print para
    #     break

    return accuracy, loss


def trainIters(utter_rnn, context_rnn, resp_rnn, train_sample, dev_sample):

    start = time.time()

    loss_track = []
    accuracy_track = []
    n_batches = len(train_sample)

    utter_rnn_optimizer = optim.SGD(utter_rnn.parameters(), lr=LEARNING_RATE)
    context_rnn_optimizer = optim.SGD(context_rnn.parameters(), lr=LEARNING_RATE)
    resp_rnn_optimizer = optim.SGD(resp_rnn.parameters(), lr=LEARNING_RATE)
    criterion = nn.NLLLoss().cuda()


    for epoch in range(1, EPOCHES + 1):
        print "epoch: %d" % epoch

        total_loss = 0
        total_accuracy = 0
        for index, sample in enumerate(train_sample):

            utter_input = sample["context"]  # context  (batch, n_prev_sent, max_n_words)
            resp_input = sample["response"]  # response
            resp_label = sample["true_res"]  # (batch, n_responses)

            accuracy, loss = train(utter_input, utter_rnn, utter_rnn_optimizer,
                  context_rnn, context_rnn_optimizer,
                  resp_input, resp_rnn, resp_rnn_optimizer,
                  resp_label, criterion)
            total_loss += loss.data[0]
            total_accuracy += accuracy

        loss_track.append(total_loss/n_batches)
        accuracy_track.append(total_accuracy/n_batches)
        print "\ttrain_set accuracy: %f, loss: %f" % (total_loss/n_batches, total_accuracy/n_batches)

        if epoch % 3 == 0:
            for index, sample in enumerate(dev_sample):
                dev_utter_input = sample["context"]  # context  (batch, n_prev_sent, max_n_words)
                dev_resp_input = sample["response"]  # response
                dev_resp_label = sample["true_res"]  # (batch, n_responses)

                dev_accu, dev_loss = train(dev_utter_input, utter_rnn, utter_rnn_optimizer,
                             context_rnn, context_rnn_optimizer,
                             dev_resp_input, resp_rnn, resp_rnn_optimizer,
                             dev_resp_label, criterion, train=False)
                print "\tdev_set accuracy: %f" % (dev_accu)

    loss_track = np.squeeze(np.asarray(loss_track))

    return loss_track



def showPlot(points):
    # plt.figure()
    # fig, ax = plt.subplots()
    # # this locator puts ticks at regular intervals
    # loc = ticker.MultipleLocator(base=0.2)
    # ax.yaxis.set_major_locator(loc)
    # plt.plot(points)

    # x = np.linspace(1, len(points), len(points))
    plt.figure()
    plt.plot(points)
    plt.show()

    # plt.figure()
    # plt.plot(points)
    # plt.show()


if __name__ == "__main__":
    vocab_words, embeddings = get_embeddings(EMBEDDING, EMBEDDING_SIZE)
    print len(vocab_words.w2i)

    train_dataset = ChatDataset("data", TRAIN_DATA)
    dev_dataset = ChatDataset("data", DEV_DATA)

    train_sample = data.DataLoader(
        dataset=train_dataset,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=2,  # 多线程来读数据
    )
    dev_sample = data.DataLoader(
        dataset=dev_dataset,  # torch TensorDataset format
        batch_size=len(dev_dataset),  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=2,  # 多线程来读数据
    )

    print "train_sample.size:"
    print len(train_sample)
    print "dev_sample.size:"
    print len(dev_sample)

    utter_rnn = rnn.RNN(embeddings, EMBEDDING_SIZE, HIDDEN_SIZE).cuda()
    context_rnn = rnn.RNN(None, HIDDEN_SIZE, HIDDEN_SIZE).cuda()
    resp_rnn = rnn.RNN(embeddings, EMBEDDING_SIZE, HIDDEN_SIZE).cuda()

    loss_track = trainIters(utter_rnn, context_rnn, resp_rnn, train_sample, dev_sample)

    torch.save(utter_rnn.state_dict(), 'model/utter_rnn_params.pkl')
    torch.save(context_rnn.state_dict(), 'model/context_rnn_params.pkl')
    torch.save(resp_rnn.state_dict(), 'model/resp_rnn_params.pkl')


    print loss_track
    # showPlot(loss_track)