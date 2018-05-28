# encoding:utf-8
import os
import time
import torch
import torch.utils.data as data
from torch import optim
from log.get_log import get_logger
from class_embedding import EmbeddingClass
from chat_dataset import ChatDataset
from class_model import GroupChatModel, GroupChatAModel_Adr
from class_loss import ModelLoss
from class_traintest import TrainTest

use_cuda = torch.cuda.is_available()

if __name__ =="__main__":
    # 0.文件路径
    root_dir = 'data'
    raw_train = os.path.join(root_dir, 'input', 'train-data.cand-2.gz')
    raw_dev = os.path.join(root_dir, 'input', 'dev-data.cand-2.gz')
    raw_test = os.path.join(root_dir, 'input', 'test-data.cand-2.gz')

    train_data = os.path.join(root_dir, 'train_cand_2.txt')
    dev_data = os.path.join(root_dir, 'dev_cand_2.txt')
    test_data = os.path.join(root_dir, 'test_cand_2.txt')

    # raw_train = os.path.join(root_dir, 'input', 'train-data-test')
    # raw_dev = os.path.join(root_dir, 'input', 'dev-data-test')
    # raw_test = os.path.join(root_dir, 'input', 'test-data-test')
    #
    # train_data = os.path.join(root_dir, 'train_cand_2_test.txt')
    # dev_data = os.path.join(root_dir, 'dev_cand_2_test.txt')
    # test_data = os.path.join(root_dir, 'dev_cand_2_test.txt')

    log_path = os.path.join('log', 'group_chat.log')
    log_name = 'group_chat'

    # 1.定义超参数
    EPOCHES = 30
    EMB_SIZE = 100
    BATCH_SIZE = 64
    HIDDEN_SIZE = 100
    LEARNING_RATE = 0.0001

    # 2.加载数据
    embedding_class = EmbeddingClass(emb_size=EMB_SIZE)
    vocab_words, word_emb = embedding_class.get_embedding(raw_train, raw_dev, raw_test)

    train_dataset = ChatDataset(train_data)
    dev_dataset = ChatDataset(dev_data)
    test_dataset = ChatDataset(test_data)
    logger = get_logger(log_path, log_name)

    train_sample = data.DataLoader(
        dataset=train_dataset,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=4,  # 多线程来读数据
    )
    # dev_sample = data.DataLoader(
    #     dataset=dev_dataset,  # torch TensorDataset format
    #     batch_size=len(dev_dataset),  # mini batch size
    #     shuffle=False,  # 要不要打乱数据 (打乱比较好)
    #     num_workers=2,  # 多线程来读数据
    # )
    test_sample = data.DataLoader(
        dataset=test_dataset,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=False,  # 要不要打乱数据 (打乱比较好)
        num_workers=4,  # 多线程来读数据
    )
    print "train_sample len:{}".format(len(train_sample))
    logger.info("train_sample len:{}".format(len(train_sample)))

    # 3.加载模型
    model = GroupChatAModel_Adr(use_cuda, word_emb, EMB_SIZE, HIDDEN_SIZE)

    # 4.定义loss和optimizer
    criterion = ModelLoss(use_cuda)
    # optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()),
    #                           lr=LEARNING_RATE, alpha=0.8, weight_decay=1e-4)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=LEARNING_RATE, weight_decay=1e-4)

    if torch.cuda.is_available():  # 判断是否有GPU加速
        model = model.cuda()
        criterion = criterion.cuda()

    # 5.加载train_test类
    train_test = TrainTest(use_cuda, model, criterion, optimizer, EPOCHES, logger)

    max_resp_acc = 0.0
    max_resp_epoch = -1
    max_spk_acc = 0.0
    max_spk_epoch = -1
    # 6.开始训练
    for epoch in range(EPOCHES):
        since = time.time()

        all_loss, all_resp_acc, all_spk_acc = train_test.train(train_sample)
        print('Finish {} epoch, Loss: {:.6f}, Resp_Acc: {:.6f}, Spk_Acc: {:.6f}, Time:{:.1f} s'.format(
            epoch + 1, all_loss, all_resp_acc, all_spk_acc, time.time() - since)
        )
        logger.info('Finish {} epoch, Loss: {:.6f}, Resp_Acc: {:.6f}, Spk_Acc: {:.6f}, Time:{:.1f} s'.format(
            epoch + 1, all_loss, all_resp_acc, all_spk_acc, time.time() - since)
        )

        if (epoch+1) % 1 == 0:
            eval_loss, eval_resp_acc, eval_spk_acc = train_test.test(test_sample)
            print('Test Loss: {:.6f}, Test Resp Acc: {:.6f}, Test Spk Acc: {:.6f}, Time:{:.1f} s'.format(
                eval_loss, eval_resp_acc, eval_spk_acc, time.time() - since)
            )
            logger.info('Test Loss: {:.6f}, Test Resp Acc: {:.6f}, Test Spk Acc: {:.6f}, Time:{:.1f} s'.format(
                eval_loss, eval_resp_acc, eval_spk_acc, time.time() - since)
            )

            if all_resp_acc > max_resp_acc:
                max_resp_acc = all_resp_acc
                max_resp_epoch = epoch + 1
            if all_spk_acc > max_spk_acc:
                max_spk_acc = all_spk_acc
                max_spk_epoch = epoch + 1

        print ("max resp Acc: {:.6f}".format(max_resp_acc))
        print ("best resp Epoch: {:d}".format(max_resp_epoch))
        print ("max spk Acc: {:.6f}".format(max_spk_acc))
        print ("best spk Epoch: {:d}".format(max_spk_epoch))

        logger.info("max resp Acc: {:.6f}".format(max_resp_acc))
        logger.info("best resp Epoch: {:d}".format(max_resp_epoch))
        logger.info("max spk Acc: {:.6f}".format(max_spk_acc))
        logger.info("best spk Epoch: {:d}".format(max_spk_epoch))


#

        # all_loss, all_acc = train_test.train(train_sample)
        # print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}, Time:{:.1f} s'.format(
        #     epoch + 1, all_loss, all_acc, time.time() - since)
        # )
        # logger.info('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}, Time:{:.1f} s'.format(
        #     epoch + 1, all_loss, all_acc, time.time() - since)
        # )
        #
        # if all_acc > max_acc:
        #     max_acc = all_acc
        #     max_epoch = epoch + 1
        #
        # if (epoch + 1) % 1 == 0:
        #     eval_loss, eval_acc, max_acc = train_test.test(test_sample)
        #     print('Test Loss: {:.6f}, Test Acc: {:.6f}, Time:{:.1f} s'.format(
        #         eval_loss, eval_acc, time.time() - since)
        #     )
        #     logger.info('Test Loss: {:.6f}, Test Acc: {:.6f}, Time:{:.1f} s'.format(
        #         eval_loss, eval_acc, time.time() - since)
        #     )
        #
        # print ("max Acc: {:.6f}".format(max_acc))
        # print ("best Epoch: {:d}".format(max_epoch))
        # logger.info("max Acc: {:.6f}".format(max_acc))
        # logger.info("best Epoch: {:d}".format(max_epoch))

#