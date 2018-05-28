# encoding:utf-8
import torch
import time
from torch.autograd import Variable
import numpy as np

class TrainTest(object):
    def __init__(self, use_cuda, model, criterion, optimizer, epoches, logger):
        super(TrainTest, self).__init__()
        self.use_cuda = use_cuda
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.max_resp_acc = 0  # test accuracy
        self.max_spk_acc = 0  # test accuracy
        self.epoches = epoches
        self.epoch = 0
        self.logger = logger


    def train(self, train_sample):
        print
        print "*" * 50
        print "epoch {}".format(self.epoch + 1)
        self.logger.info('')
        self.logger.info("*" * 50)
        self.logger.info("epoch {}".format(self.epoch + 1))

        total_loss = 0.0
        total_resp_correct = 0.0
        total_spk_correct = 0.0
        total_sample = 0
        self.model.train()
        self.optimizer.zero_grad()

        for index, sample in enumerate(train_sample):
            since = time.time()
            context = sample["context"]  # context  (batch, n_prev_sent, max_n_words)
            response = sample["response"]  # response  (batch, n_response, max_n_word)
            resp_label = sample["resp_label"]  # (batch)  1 or -1
            resp_target = sample["true_res"]
            spk_agents = sample["spk_agents"]  # spk_agents  (batch, n_prev_sent)
            spk_target = sample["true_adr"]

            batch_size = len(context)
            total_sample += batch_size

            if self.use_cuda:
                context = Variable(context).cuda()
                response = Variable(response).cuda()
                resp_label = Variable(resp_label).cuda()
                spk_agents = Variable(spk_agents).cuda()
                spk_target = Variable(spk_target).cuda()
                spk_label = Variable(torch.ones(batch_size)).cuda()
            else:
                context = Variable(context)
                response = Variable(response)
                resp_label = Variable(resp_label)
                spk_agents = Variable(spk_agents)
                spk_target = Variable(spk_target)
                spk_label = Variable(torch.ones(batch_size))

            score_r, score_s = self.model(context, response, resp_target, spk_agents, spk_target)  # (2, batch) (batch, n_prev_sents)
            loss = self.criterion(score_r, score_s, resp_label.float(), spk_label)

            total_loss += loss.item() * resp_label.size(0)
            _, resp_predict = torch.max(score_r, 0)  # (batch,)
            resp_predict = resp_predict.data.cpu()
            _, spk_predict = torch.max(score_s, 0)
            # spk_predict = self.get_predict_spk(spk_predict, spk_agents)  # (batch)

            resp_correct = (resp_predict == resp_target).sum()
            total_resp_correct += resp_correct
            spk_correct = (spk_predict.data == spk_target.data).sum()  # spk_correct = (spk_predict == spk_target.data).sum()
            total_spk_correct += spk_correct

            # print "total_resp_correct:{}".format(total_resp_correct.item())
            # print "total_spk_correct:{}".format(total_spk_correct.item())

            running_loss = total_loss / (index + 1)
            running_resp_acc = total_resp_correct.item() / float((index + 1) * batch_size)
            running_spk_acc = total_spk_correct.item() / float((index + 1) * batch_size)

            # 向后传播
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # for name, para in self.model.named_parameters():
            #     if name.startswith("atten_model"):
            #         print name
            #         print para

            if (index+1) % 100 == 0:
                print('[{}/{}] Train Loss: {:.6f}, Time:{:.1f} s'.format(
                    self.epoch + 1, self.epoches, running_loss, time.time() - since)
                )
                print('resp_accuracy: {:.6f}, spk_accuracy: {:.6f}'.format(
                    running_resp_acc, running_spk_acc)
                )
                self.logger.info('[{}/{}] Train Loss: {:.6f}, Time:{:.1f} s'.format(
                    self.epoch + 1, self.epoches, running_loss, time.time() - since)
                )
                self.logger.info('resp_accuracy: {:.6f}, spk_accuracy: {:.6f}'.format(
                    running_resp_acc, running_spk_acc)
                )

        self.epoch += 1
        all_loss = 1.0 * running_loss
        all_resp_acc = 1.0 * total_resp_correct.item() / total_sample  # 总共对的个数/总共的sample数
        all_spk_acc = 1.0 * total_spk_correct.item() / total_sample  # 总共对的个数/总共的sample数

        return all_loss, all_resp_acc, all_spk_acc


    def get_predict_spk(self, spk_predict, spk_agents):
        # spk_predict:(batch)
        # spk_agents: (batch, n_prev_sent)

        true_spk = Variable(torch.zeros(*spk_predict.size()))
        if self.use_cuda:
            true_spk = true_spk.cuda()
        for i,  spk_index in enumerate(spk_predict):
            spk_index = spk_index.data
            true_spk[i] = spk_agents[i][spk_index]

        return true_spk.long().data


    def test(self, test_sample):
        self.model.eval()
        eval_loss = 0.0
        total_resp_correct = 0.0
        total_spk_correct = 0.0
        total_sample = 0

        for index, sample in enumerate(test_sample):
            context = sample["context"]  # context  (batch, n_prev_sent, max_n_words)
            response = sample["response"]  # response  (batch, n_response, max_n_word)
            resp_label = sample["resp_label"]  # (batch)  1 or -1
            resp_target = sample["true_res"]
            spk_agents = sample["spk_agents"]  # spk_agents  (batch, n_prev_sent)
            spk_target = sample["true_adr"]
            batch_size = len(context)
            total_sample += batch_size

            with torch.no_grad():
                if self.use_cuda:
                    context = Variable(context).cuda()
                    response = Variable(response).cuda()
                    resp_label = Variable(resp_label).cuda()
                    spk_agents = Variable(spk_agents).cuda()
                    spk_target = Variable(spk_target).cuda()
                    spk_label = Variable(torch.ones(batch_size)).cuda()
                else:
                    context = Variable(context)
                    response = Variable(response)
                    resp_label = Variable(resp_label)
                    spk_agents = Variable(spk_agents)
                    spk_target = Variable(spk_target)
                    spk_label = Variable(torch.ones(batch_size))

            # score = self.model(context, response)
            # loss = self.criterion(score[0], score[1], resp_label.float())

            score_r, score_s = self.model(context, response, resp_target, spk_agents, spk_target)  # (2, batch) (batch, n_prev_sents)
            loss = self.criterion(score_r, score_s, resp_label.float(), spk_label)

            # eval_loss += loss.data[0] * resp_label.size(0)
            # _, resp_predict = torch.max(score, 0)  # (batch,)
            # resp_predict = resp_predict.data.cpu()
            # num_correct = (resp_predict == resp_target).sum()
            # total_correct += num_correct

            eval_loss += loss.item() * resp_label.size(0)
            _, resp_predict = torch.max(score_r, 0)  # (batch,)
            resp_predict = resp_predict.data.cpu()
            _, spk_predict = torch.max(score_s, 0)
            # spk_predict = self.get_predict_spk(spk_predict, spk_agents)  # (batch)

            resp_correct = (resp_predict == resp_target).sum()
            total_resp_correct += resp_correct
            spk_correct = ( spk_predict.data == spk_target.data).sum()
            total_spk_correct += spk_correct

        eval_loss = 1.0 * eval_loss / len(test_sample)
        eval_resp_acc = 1.0 * total_resp_correct.item() / total_sample
        eval_spk_acc = 1.0 * total_spk_correct.item() / total_sample

        return eval_loss, eval_resp_acc, eval_spk_acc







