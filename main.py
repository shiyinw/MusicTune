from __future__ import print_function
from math import ceil
import numpy as np
import sys
import pdb, pickle

import torch, datetime
import torch.optim as optim
import torch.nn as nn

import generator
import discriminator
import helpers
from oracle import Dataloader

import random, time



CUDA = torch.cuda.is_available()
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
DIS_HIDDEN_DIM = 32

log = []

def split_data(batch, bs):
    inp = torch.autograd.Variable(torch.zeros(bs, MAX_SEQ_LEN)).type(torch.LongTensor)
    target = torch.autograd.Variable(torch.zeros(bs, MAX_SEQ_LEN)).type(torch.LongTensor)
    for i in range(min(bs, len(batch))):
        inp[i, :] = torch.from_numpy(batch[i][0][:MAX_SEQ_LEN])
        target[i, :] = torch.from_numpy(batch[i][1][:MAX_SEQ_LEN])
    return inp, target


def train_generator_MLE(gen, gen_opt, oracle, dataloader, epochs):
    """
    Max Likelihood Pretraining for the generator
    """
    for epoch in range(epochs):
        start_time = time.time()
        print('epoch %d/%d : ' % (epoch + 1, epochs), end='')
        sys.stdout.flush()
        total_loss = 0

        select_samples = dataloader.train

        for i in range(0, POS_NEG_SAMPLES, BATCH_SIZE):

            batch = select_samples[i:i + BATCH_SIZE]
            inp, target = split_data(batch, BATCH_SIZE)

            gen_opt.zero_grad()
            loss = gen.batchNLLLoss(inp, target)
            loss.backward()
            gen_opt.step()

            total_loss += loss.data.item()

            if (i / BATCH_SIZE) % ceil(
                            ceil(POS_NEG_SAMPLES / float(BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
                print('.', end='')
                sys.stdout.flush()

        # each loss in a batch is loss per sample
        total_loss = total_loss / ceil(POS_NEG_SAMPLES / float(BATCH_SIZE)) / MAX_SEQ_LEN

        # sample from generator and compute oracle NLL
        oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN,
                                                   start_letter=START_LETTER, gpu=CUDA)

        msg = ' average_train_NLL = %.4f, oracle_sample_NLL = %.4f, time = %.2f' % (total_loss, oracle_loss, time.time()-start_time)
        log.append(msg)
        print(msg)


def train_generator_PG(gen, gen_opt, oracle, dis, num_batches, dataloader):
    """
    The generator is trained using policy gradients, using the reward from the discriminator.
    Training is done for num_batches batches.
    """

    start_time = time.time()

    for batch in range(num_batches):
        inp, _ = dataloader.sample_pos(64)

        hidden = gen.init_hidden(MAX_SEQ_LEN)
        target, _ = gen(inp, hidden)

        # s = gen.sample(BATCH_SIZE*2)        # 64 works best
        # inp, target = helpers.prepare_generator_batch(s, start_letter=START_LETTER, gpu=CUDA)
        rewards = dis.batchClassify(target)

        gen_opt.zero_grad()
        pg_loss = gen.batchPGLoss(inp, target, rewards)
        print(batch, pg_loss.data[0])
        pg_loss.backward()
        gen_opt.step()


def train_discriminator(discriminator, dis_opt, pos_samples, neg_samples, generator, oracle, d_steps, epochs, dataloader):
    """
    Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
    Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
    """

    # generating a small validation set before training (using oracle and generator)

    pos_val, neg_val = dataloader.sample_valid(100)
    val_inp, val_target = helpers.prepare_discriminator_data(pos_val, neg_val, gpu=CUDA)

    for d_step in range(d_steps):
        s = helpers.batchwise_sample(generator, POS_NEG_SAMPLES, BATCH_SIZE)
        pos_samples = dataloader.sample_pos(POS_NEG_SAMPLES)

        hidden = gen.init_hidden(MAX_SEQ_LEN)
        neg_samples, _ = gen(pos_samples, hidden)
        pos_samples = dataloader.sample_pos(POS_NEG_SAMPLES)
        dis_inp, dis_target = helpers.prepare_discriminator_data(pos_samples, neg_samples, gpu=CUDA)

        for epoch in range(epochs):
            msg = 'd-step %d/%d epoch %d/%d : ' % (d_step + 1, d_steps, epoch + 1, epochs)
            log.append(msg)
            print(msg)

            sys.stdout.flush()
            total_loss = 0
            total_acc = 0

            for i in range(0, 2 * POS_NEG_SAMPLES, BATCH_SIZE):
                inp, target = dis_inp[i:i + BATCH_SIZE], dis_target[i:i + BATCH_SIZE]

                dis_opt.zero_grad()
                out = discriminator.batchClassify(inp)
                loss_fn = nn.BCELoss()
                loss = loss_fn(out, target)
                loss.backward()
                dis_opt.step()

                total_loss += loss.data.item()
                total_acc += torch.sum((out>0.5)==(target>0.5)).data.item()

                if (i / BATCH_SIZE) % ceil(ceil(2 * POS_NEG_SAMPLES / float(
                        BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
                    print('.', end='')
                    sys.stdout.flush()

            total_loss /= ceil(2 * POS_NEG_SAMPLES / float(BATCH_SIZE))
            total_acc /= float(2 * POS_NEG_SAMPLES)

            val_pred = discriminator.batchClassify(val_inp)
            msg = ' average_loss = %.4f, train_acc = %.4f, val_acc = %.4f' % (
                total_loss, total_acc, torch.sum((val_pred>0.5)==(val_target>0.5)).data.item()/200.)

            log.append(msg)
            print(msg)


# MAIN
if __name__ == '__main__':

    curtime = datetime.datetime.now().strftime("%I:%M%p_%B_%d")
    oracle = generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA, oracle_init=True)
    #oracle.load_state_dict(torch.load(oracle_state_dict_path))

    # a new oracle can be generated by passing oracle_init=True in the generator constructor
    # samples for the new oracle can be generated using helpers.batchwise_sample()

    gen = generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)
    dis = discriminator.Discriminator(DIS_EMBEDDING_DIM, DIS_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)


    with open("./data.pickle", "rb") as f:
        samples = pickle.load(f)

    param = {"cuda": CUDA, "vocal_size": VOCAB_SIZE, "max_seq_len": MAX_SEQ_LEN, "start_letter": START_LETTER,
             "batch_size": BATCH_SIZE, "mle_train_epochs": MLE_TRAIN_EPOCHS,
             "adv_train_epochs": ADV_TRAIN_EPOCHS, "pos_neg_samples": POS_NEG_SAMPLES,
             "gen_embedding_dim": GEN_EMBEDDING_DIM, "gen_hidden_dim": GEN_EMBEDDING_DIM,
             "dis_embedding_dim": DIS_EMBEDDING_DIM, "dis_hidden_dim": DIS_HIDDEN_DIM}

    print(param)

    with open(curtime + ".config", "w") as f:
        f.write("\n".join(log))
        for k in param.keys():
            f.write(k)
            f.write(" = ")
            f.write(str(param[k]))
            f.write('\n')


    np.random.shuffle(samples)
    num = len(samples)


    pos_samples, neg_samples = split_data(samples, len(samples))


    n_train = int(num*0.8)
    n_vaild = int(num*0.9) # start index of validation set

    dataloader = Dataloader(samples[:n_train])

    # GENERATOR MLE TRAINING
    msg = 'Starting Generator MLE Training...'
    log.append(msg)
    print(msg)

    gen_optimizer = optim.Adam(gen.parameters(), lr=1e-2)


    # train_generator_MLE(gen, gen_optimizer, oracle, dataloader, MLE_TRAIN_EPOCHS)
    # torch.save(gen.state_dict(), "pretrained_gen.model")

    if CUDA:
        gen.load_state_dict(torch.load("pretrained_gen.model"))
    else:
        params = torch.load("pretrained_gen.model", map_location="cpu")
        gen.load_state_dict(params)

    # PRETRAIN DISCRIMINATOR
    msg = '\nStarting Discriminator Training...'
    log.append(msg)
    print(msg)

    dis_optimizer = optim.Adam(dis.parameters())

    train_discriminator(dis, dis_optimizer, pos_samples, neg_samples, gen, oracle, 10, 3, dataloader)  # 50, 3
    torch.save(dis.state_dict(), "pretrained_dis.model")

    # if CUDA:
    #     dis.load_state_dict(torch.load("pretrained_dis.model"))
    # else:
    #     params = torch.load("pretrained_dis.model", map_location="cpu")
    #     dis.load_state_dict(params)

    # ADVERSARIAL TRAINING
    msg = '\nStarting Adversarial Training...'
    log.append(msg)
    print(msg)

    for epoch in range(ADV_TRAIN_EPOCHS):
        msg = '\n--------\nEPOCH %d\n--------' % (epoch+1)
        log.append(msg)
        print(msg)

        # TRAIN GENERATOR
        msg = '\nAdversarial Training Generator : '
        log.append(msg)
        print(msg)

        sys.stdout.flush()
        train_generator_PG(gen, gen_optimizer, oracle, dis, 10, dataloader)

        # TRAIN DISCRIMINATOR
        msg = '\nAdversarial Training Discriminator : '
        log.append(msg)
        print(msg)

        train_discriminator(dis, dis_optimizer, pos_samples, neg_samples, gen, oracle, 2, 3, dataloader)

        torch.save("./MusicTune/" + dis.state_dict(), curtime + "_epoch_" + str(epoch) + "_dis.model")
        torch.save("./MusicTune/" + gen.state_dict(), curtime + "_epoch_" + str(epoch) + "_gen.model")

    with open(curtime+"_log.txt", "w") as f:
        f.write("\n".join(log))

    torch.save(dis.state_dict(), curtime+"_dis.model")
    torch.save(gen.state_dict(), curtime+"_gen.model")





