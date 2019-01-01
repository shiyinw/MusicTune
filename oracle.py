import torch
import random
import numpy as np


MAX_SEQ_LEN = 3000

def split_data(batch, bs):
    inp = torch.autograd.Variable(torch.zeros(bs, MAX_SEQ_LEN)).type(torch.LongTensor)
    target = torch.autograd.Variable(torch.zeros(bs, MAX_SEQ_LEN)).type(torch.LongTensor)
    for i in range(min(bs, len(batch))):
        inp[i, :] = torch.from_numpy(batch[i][0])
        target[i, :] = torch.from_numpy(batch[i][1])
    return inp, target

class Dataloader:
    def __init__(self, data):
        self.num = len(data)

        np.random.shuffle(data)

        n_train = int(self.num * 0.8)
        n_vaild = int(self.num * 0.9)  # start index of validation set

        self.train = data[:n_train]
        self.pos_train = split_data(data[:n_train], n_train)
        self.pos_test, self.neg_test = split_data(data[n_train:n_vaild], n_vaild - n_train)
        self.pos_valid, self.neg_vaild = split_data(data[n_vaild:], self.num - n_vaild)

        self.n_train = n_train
        self.n_test = n_vaild - n_train
        self.n_valid = self.num - n_vaild

        self.valid = data[n_vaild:]

    def sample_valid(self, num):
        sample = random.sample(self.valid, num)
        pos, neg = split_data(sample, num)
        return pos, neg

    def sample_train(self, num):
        sample = random.sample(self.train, num)
        pos, neg = split_data(sample, num)
        return pos, neg

    def sample_pos(self, num):
        pos, neg = self.sample_train(num)
        return pos

    def sample_neg(self, num):
        pos, neg = self.sample_train(num)
        return neg