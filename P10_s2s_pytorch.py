import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader, sampler
from pytorchtools import EarlyStopping, Encoder, Decoder, InferenceDecoder

torch.manual_seed(1)

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print(device)

PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2

##############################################
############ Data Preprocessing ##############
##############################################
class BuildVocab:
    def __init__(self):
        self.word2index = {'<PAD>': PAD_TOKEN, '<SOS>': SOS_TOKEN, '<EOS>': EOS_TOKEN}
        self.word2count = {}
        self.index2word = {PAD_TOKEN:'<PAD>', SOS_TOKEN: '<SOS>', EOS_TOKEN: '<EOS>'}
        self.num_words = 3
    
    def fromCorpus(self, corpus):
        for line in corpus:
            self.fromSentence(line)
    
    def fromSentence(self, line):
        for word in line:
            self.toVocab(word)
    
    def toVocab(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

# mapping character to index for all sentences
def map_character(voc, line):
    return [voc.word2index[word] for word in line] + [EOS_TOKEN] # EOS token

def sequence_padding(ctx, max_len):
    for index, seq in enumerate(ctx):
        pad_to_seq = [PAD_TOKEN]*(max_len - len(seq))
        ctx[index] = [*seq, *pad_to_seq]
    return ctx

def target_binary_mask(ctx):
    for i, line in enumerate(ctx):
        for j, index in enumerate(line):
            if index != 0:
                ctx[i][j] = 1
    return ctx

# input and output batch shape should be (max_length, batch_size)
def input_init(input_batch, voc):
    mapped_batch = [map_character(voc, line) for line in input_batch]
    lengths = [len(seq_len) for seq_len in mapped_batch]
    length_t = torch.tensor(lengths)
    max_input_len = max(lengths)
    padded_batch = sequence_padding(mapped_batch, max_input_len)
    padded_batch_t = torch.LongTensor(padded_batch).t()
    return padded_batch_t, length_t

def target_init(target_batch, voc):
    mapped_batch = [map_character(voc, line) for line in target_batch]
    max_target_len = max(len(seq_len) for seq_len in mapped_batch)
    padded_batch = sequence_padding(mapped_batch, max_target_len)
    padded_batch_t = torch.LongTensor(padded_batch).t()
    target_mask = target_binary_mask(padded_batch)
    target_mask_t = torch.ByteTensor(target_mask).t()
    return max_target_len, padded_batch_t, target_mask_t

def batch_inp_out(loader):
    batches = []
    for input_batch, target_batch in loader:
        # sort lengths to pack_padded_sequence later
        input_batch, target_batch = zip(*sorted(zip(input_batch, target_batch),
                                                key=lambda x:len(x[0]),
                                                reverse=True))
        input_batch = list(input_batch)
        target_batch = list(target_batch)
        input_t, lengths = input_init(input_batch, voc)
        max_target_len, target_t, mask_t = target_init(target_batch, voc)
        batches.append((input_t, lengths, max_target_len, target_t, mask_t))
    return batches

def train_val_split(input_seq, target_seq, batch_size):
    num_train = len(input_seq)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(0.1 * num_train)) # validate on 10% of data
    train_indx, val_indx = indices[split:], indices[:split]
    
    data = list(zip(input_seq, target_seq))
    train_sampler = sampler.SubsetRandomSampler(train_indx)
    val_sampler = sampler.SubsetRandomSampler(val_indx)
    
    train_loader = DataLoader(data, batch_size=batch_size, sampler=train_sampler, drop_last=True)
    val_loader = DataLoader(data, batch_size=batch_size, sampler=val_sampler, drop_last=True)
    
    train_batches = batch_inp_out(train_loader)
    val_batches = batch_inp_out(val_loader)
    return train_batches, val_batches

# calculate loss and ignore padded part
# dec_out: (batch_size, vocab_size)
def loss_func(dec_out, target, mask):
    # (batch_size, 1)
    target = target.view(-1, 1)
    # negative log likelihood
    crossEntropy = -torch.log(torch.gather(dec_out, 1, target).squeeze(1))
    # select non-zero elements and calculate the mean
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, mask.sum().item()


########################################################
#################### Model Traning #####################
########################################################
# calculate loss based on teacher forcing
def loss_calulation(encoder, decoder, batch, batch_size, teacher_forcing_r):
    
    input_t, lengths, max_target_len, target_t, mask_t = batch
    
    input_t = input_t.to(device)
    lengths = lengths.to(device)
    target_t = target_t.to(device)
    mask_t = mask_t.to(device)
    
    encoder_output, encoder_hidden = encoder(input_t, lengths)
    decoder_input = torch.LongTensor([[SOS_TOKEN for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)
    # set initial decoder hidden state to the encoder's final state
    decoder_hidden = encoder_hidden[:decoder.n_layer]
    
    loss = 0
    print_loss = []
    nTotals = 0
    if np.random.random() < teacher_forcing_r:
        for step in range(max_target_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)
            # teacher forcing: next input is current target
            decoder_input = target_t[step].view(1, -1)
            mask_loss, nTotal = loss_func(decoder_output, target_t[step], mask_t[step])
            loss += mask_loss
            print_loss.append(mask_loss.item()*nTotal)
            nTotals += nTotal
    else:
        for step in range(max_target_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)
            # non teacher forcing: next input is current decoder output
            # choose the index of word with highest possibility
            _, indx = torch.max(decoder_output, dim=1) 
            decoder_input = torch.LongTensor([[indx[i] for i in range(batch_size)]])
            decoder_input.to(device)
            mask_loss, nTotal = loss_func(decoder_output, target_t[step], mask_t[step])
            loss += mask_loss
            print_loss.append(mask_loss.item()*nTotal)
            nTotals += nTotal
    
    return print_loss, loss, nTotals

# training process wrap
def train(input_seq, target_seq, batch_size, epochs, hidden_dim, embedding, encoder, decoder,
          encoder_opt, decoder_opt, teacher_forcing_r, epoch_from, folder, patience=20):
    
    avg_train_loss = []
    avg_val_loss = []
    
    train_batches, val_batches = train_val_split(input_seq, target_seq, batch_size)
    
    print(encoder)
    print(decoder)
    
    early_stopping = EarlyStopping(patience=patience)
    
    for epoch in range(epoch_from, epochs):

        ############
        # Training #
        ############
        encoder.train()
        decoder.train()
        train_loss = []
        for train_batch in train_batches:
            # initialize and move the model to GPU/CPU
            # set training mode
            encoder_opt.zero_grad()
            decoder_opt.zero_grad()

            print_loss, loss, nTotals = loss_calulation(encoder, decoder, train_batch, batch_size, teacher_forcing_r)

            loss.backward()

            _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)

            encoder_opt.step()  # update weights
            decoder_opt.step()
            
            train_loss.append(sum(print_loss) / nTotals)
        
        print("train_finished!")

        ##############
        # validation #
        ##############
        encoder.eval()
        decoder.eval()
        val_loss = []
        for val_batch in val_batches:
            print_loss, _, nTotals = loss_calulation(encoder, decoder, val_batch, batch_size, teacher_forcing_r)
            
            val_loss.append(sum(print_loss) / nTotals)  # loss for batches

        avg_train_loss.append(np.average(train_loss))  # loss for epochs
        avg_val_loss.append(np.average(val_loss))
        print("Epoch: {}; train loss: {:.4}; val loss: {:.4}".format(epoch + 1, avg_train_loss[-1], avg_val_loss[-1]))
        
        model_states = (
            epoch,
            encoder.state_dict(),
            decoder.state_dict(),
            encoder_opt.state_dict(),
            decoder_opt.state_dict(),
            voc.__dict__,
            embedding.state_dict())
        
        early_stopping(avg_val_loss[-1], model_states, folder)
        
        if early_stopping.early_stop:
            print('EarlyStopping ...')
            break
        
    return avg_train_loss, avg_val_loss

def run(folder, loadMode):
    if loadMode:
        checkpoint = torch.load(loadMode)
        enc_chp = checkpoint["enc"]
        dec_chp = checkpoint["dec"]
        enc_opt_chp = checkpoint["enc_opt"]
        dec_opt_chp = checkpoint["dec_opt"]
        voc.__dict__ = checkpoint["voc_dict"]
        embedding_chp = checkpoint["embedding"]

    embedding = nn.Embedding(voc.num_words, hidden_dim)
    if loadMode:
        embedding.load_state_dict(embedding_chp)

    encoder = Encoder(hidden_dim, embedding, enc_layer_dim, dropout)
    decoder = Decoder(attn_method, embedding, hidden_dim, voc.num_words, dec_layer_dim, dropout)
    if loadMode:
        encoder.load_state_dict(enc_chp)
        decoder.load_state_dict(dec_chp)

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    encoder_opt = optim.Adam(encoder.parameters(), lr=lr)
    decoder_opt = optim.Adam(decoder.parameters(), lr=lr)

    epoch_from = 0
    if loadMode:
        encoder_opt.load_state_dict(enc_opt_chp)
        decoder_opt.load_state_dict(dec_opt_chp)
        epoch_from = checkpoint['epoch'] + 1

    train_loss, val_loss = train(input_seq, target_seq, batch_size, epochs, hidden_dim, embedding,
                                 encoder, decoder, encoder_opt, decoder_opt, teacher_forcing_r,
                                 epoch_from, folder)
    
    with open(os.path.join(folder, 'loss.txt'), 'w') as f:
        for loss1, loss2 in list(zip(train_loss, val_loss)):
            f.write("{}\t{}\n".format(loss1, loss2))

def evaluation(inference, voc, test_seq, chop_size):    
    # word2index
    mapped_batch = [map_character(voc, test_seq)]
    lengths = torch.tensor([len(mapped_batch[0])])
    input_batch = torch.LongTensor(mapped_batch).t()
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    preds, probs = inference(input_batch, lengths, chop_size)
    decoded_words = [voc.index2word[indx.item()] for indx in preds]
    return decoded_words

def translate(loadMode, rand_indx, chop_size):
    checkpoint = torch.load(loadMode)
    enc_chp = checkpoint["enc"]
    dec_chp = checkpoint["dec"]
    voc.__dict__ = checkpoint["voc_dict"]
    embedding_chp = checkpoint["embedding"]

    embedding = nn.Embedding(voc.num_words, hidden_dim)
    embedding.load_state_dict(embedding_chp)

    encoder = Encoder(hidden_dim, embedding, enc_layer_dim, dropout)
    decoder = Decoder(attn_method, embedding, hidden_dim, voc.num_words, dec_layer_dim, dropout)
    encoder.load_state_dict(enc_chp)
    decoder.load_state_dict(dec_chp)

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    
    encoder.eval()
    decoder.eval()

    test_seq = input_seq[rand_indx]
    test_tar = target_seq[rand_indx]
    inference = InferenceDecoder(encoder, decoder)
    decoded_words = evaluation(inference, voc, test_seq, chop_size)
    print("Input: ", test_seq)
    print("Target: ", test_tar)
    print("Prediction:", ''.join(decoded_words))


##############################################
################# Run Model ##################
##############################################

# hyperparameters
hidden_dim = 500
enc_layer_dim = 3
dec_layer_dim = 3
dropout = 0.1
attn_method = "dot"
lr = 0.0001
teacher_forcing_r = 0.5
batch_size = 128
epochs = 2000
chop_size = 500  # keep sentence length <= chop_size

data_file = "blastp_best_result/protein_seq.csv"

if __name__ == "__main__":
    df = pd.read_csv(data_file, index_col=0)
    df.drop_duplicates(inplace=True)
    print("Trimming sentence from: ", df.shape)

    # chop off the sequence length
    # keep sentence length <= 500
    mask = (df["meso_seq"].str.len()<=chop_size)&(df["thermal_seq"].str.len()<=chop_size)
    df = df.loc[mask].reset_index(drop=True)
    print("To: ", df.shape)

    input_seq = df["meso_seq"].tolist()
    target_seq = df["thermal_seq"].tolist()

    voc = BuildVocab()
    voc.fromCorpus(input_seq + target_seq)
    print("Word Count: ", len(voc.word2index))

    ####### training process ##########
    ## training process
    folder = os.path.join("checkpoints", "{}_{}_{}".format(enc_layer_dim, dec_layer_dim, hidden_dim))
    loadMode = None

    if not os.path.exists(folder):
        os.makedirs(folder)

    run(folder, loadMode)

    ## translation process
    seq_indx = np.random.choice(range(len(input_seq)))
    loadMode = os.path.join(folder, "checkpoint.tar")
    translate(loadMode, seq_indx, chop_size)

