from __future__ import print_function
from math import ceil
import numpy as np
import sys
import pdb, pickle

import torch
import torch.optim as optim
import torch.nn as nn

import generator
import discriminator
import helpers

CUDA = False
VOCAB_SIZE = 1000 # pitch frequency
MAX_SEQ_LEN = 200
START_LETTER = 0
BATCH_SIZE = 32 # 32
MLE_TRAIN_EPOCHS = 20 # 100
ADV_TRAIN_EPOCHS = 30 # 50
POS_NEG_SAMPLES = 320 # number of samples

# total number 34008

GEN_EMBEDDING_DIM = 10
GEN_HIDDEN_DIM = 100
DIS_EMBEDDING_DIM = 10
DIS_HIDDEN_DIM = 64


def split_data(batch, bs):
    inp = torch.autograd.Variable(torch.zeros(bs, MAX_SEQ_LEN)).type(torch.LongTensor)
    target = torch.autograd.Variable(torch.zeros(bs, MAX_SEQ_LEN)).type(torch.LongTensor)
    for i in range(min(bs, len(batch))):
        inp[i, :] = torch.from_numpy(batch[i][0][:min(MAX_SEQ_LEN, len(batch[i][0]))])
        target[i, :] = torch.from_numpy(batch[i][1][:min(MAX_SEQ_LEN, len(batch[i][1]))])
    return inp, target


def prepare_seq(seq):
    inp = torch.autograd.Variable(torch.zeros(1, MAX_SEQ_LEN)).type(torch.LongTensor)
    L = seq.shape()
    inp[0, :] = seq[:min(MAX_SEQ_LEN, L)]
    return inp


if __name__ == "__main__":

    oracle = generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA, oracle_init=True)
    gen = generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)
    dis = discriminator.Discriminator(DIS_EMBEDDING_DIM, DIS_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)

    gpu_gen = generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=True)
    params = torch.load("07:04PM_January_01_epoch_5_gen.model", map_location="cpu")
    #gpu_gen.load_state_dict(torch.load("gen.model"))
    gen.load_state_dict(params)
    print(gen)
    #gen.load_state_dict(torch.load("gen.model"))

    with open("./pitch_data/0.pickle", "rb") as f:
        samples = pickle.load(f)

    if CUDA:
        oracle = oracle.cuda()
        gen = gen.cuda()
        dis = dis.cuda()
        samples = samples.cuda()

    pos_samples, neg_samples = split_data(samples, len(samples))

    input = pos_samples[1:2, :][:MAX_SEQ_LEN]
    print(input)

    hidden = gen.init_hidden(MAX_SEQ_LEN)
    predict, _ = gen(input, hidden)

    print(predict.size())
    values, indices = torch.max(predict, 1)

    print(indices)