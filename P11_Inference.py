import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
from pytorchtools import Encoder, Decoder, InferenceDecoder
from P10_s2s_pytorch import BuildVocab, map_character

torch.manual_seed(1)

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print(device)

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

def translate(loadMode, test_seq, test_tar, chop_size, voc):
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

    inference = InferenceDecoder(encoder, decoder, device)
    decoded_words = evaluation(inference, voc, test_seq, chop_size)
    print("Input: ", test_seq)
    print("Target: ", test_tar)
    print("Prediction:", ''.join(decoded_words))

hidden_dim = 300
enc_layer_dim = 2
dec_layer_dim = 2
dropout = 0.1
attn_method = "general"
lr = 0.0001
teacher_forcing_r = 0.5
batch_size = 64
epochs = 500
chop_size = 100

## translation process
chop_size=200
voc = BuildVocab()
folder = "checkpoints/general_128_2_0.1_0.001_300_200"
df = pd.read_csv(folder + "/test.txt", sep=',', header=None)
test_seq = df.iloc[:, :1].values.flatten().tolist()
test_tar = df.iloc[:, 1:].values.flatten().tolist()
seq_indx = np.random.choice(range(len(test_seq)))
test_seq0 = test_seq[seq_indx]
test_tar0 = test_tar[seq_indx]

loadMode = folder + "/checkpoint.tar"
translate(loadMode, test_seq0, test_tar0, chop_size, voc)
# 
# loadMode = os.path.join(folder, "checkpoint.tar")
# translate(loadMode, seq_indx, chop_size)
