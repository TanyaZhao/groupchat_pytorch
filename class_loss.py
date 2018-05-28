# encoding:utf-8
from torch import nn

class ModelLoss(nn.Module):
    def __init__(self, cuda):
        super(ModelLoss,self).__init__()
        self.criterion1 = nn.MarginRankingLoss(margin=0.2)
        # self.criterion2 = nn.NLLLoss()
        self.criterion2 = nn.MarginRankingLoss(margin=0.2)
        if cuda:
            self.criterion1 = self.criterion1.cuda()
            self.criterion2 = self.criterion2.cuda()
        # self.criterion1 = nn.CrossEntropyLoss()
        # self.criterion2 = nn.MarginRankingLoss()

    def forward(self, resp_score, spk_score, resp_label, spk_label):
        loss_resp = self.criterion1(resp_score[0], resp_score[1], resp_label)
        loss_spk = self.criterion2(spk_score[0], spk_score[1], spk_label)
        return loss_resp + loss_spk