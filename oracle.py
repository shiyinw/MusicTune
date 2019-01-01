import torch
import random


MAX_SEQ_LEN = 3000

def split_data(batch, bs):
    inp = torch.autograd.Variable(torch.zeros(bs, MAX_SEQ_LEN)).type(torch.LongTensor)
    target = torch.autograd.Variable(torch.zeros(bs, MAX_SEQ_LEN)).type(torch.LongTensor)
    for i in range(min(bs, len(batch))):
        inp[i, :] = torch.from_numpy(batch[i][0])
        target[i, :] = torch.from_numpy(batch[i][1])
    return inp, target

class oracle:
    def __init__(self, data):
        self.data = data

    def sample_pos(self, num):
        sample = random.sample(self.data, num)
        pos, _ = split_data(sample)
        return pos

    def sample_neg(self, num):
        sample = random.sample(self.data, num)
        _, neg = split_data(sample)
        return neg