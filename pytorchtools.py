import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def stop_until_epochs(self, val_loss, model_states, folder, epochs):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model_states, folder)
        elif score > self.best_score:
            self.best_score = score
            self.save_checkpoint(val_loss, model_states, folder)
        
        self.counter += 1

        if self.counter >= epochs:
            self.early_stop = True

    def stop_early(self, val_loss, model_states, folder):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model_states, folder)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model_states, folder)
            self.counter = 0

    def save_checkpoint(self, val_loss, model_states, folder):
        '''Saves model when validation loss decrease.'''
        epoch, enc_st, dec_st, enc_opt_st, dec_opt_st, voc_st, em_st = model_states
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save({
            'epoch': epoch,
            'enc': enc_st,
            'dec': dec_st,
            'enc_opt': enc_opt_st,
            'dec_opt': dec_opt_st,
            'voc_dict': voc_st,
            'embedding': em_st
        }, os.path.join(folder, 'checkpoint.tar'))
        self.val_loss_min = val_loss


# network inherit from nn.Module (container) and override forward() method
class Encoder(nn.Module):
    def __init__(self, hidden_dim, embedding, layer_dim, dropout=0):
        super(Encoder, self).__init__()
        self.layer_dim, self.hidden_dim, self.embedding = layer_dim, hidden_dim, embedding
        # nn.GRU(input_dim, hidden_dim, layer_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, layer_dim, dropout=dropout, bidirectional=True)
    
    # input_seq: (max_length, b atch_size)
    # input_length: (batch_size)
    def forward(self, input_seq, input_length):
        
        # for each batch_size add a random feature with shape of embedding_dim
        # embeddings: (max_length, batch_size, embedding_dim=hidden_dim)
        embeddings = self.embedding(input_seq)
        
        # tensor will be concated in time axis
        # basically, it will again concat the first column (seq), second, ....
        # pack: (sum_seq_len, num_directions * hidden_size)
        pack = torch.nn.utils.rnn.pack_padded_sequence(embeddings, input_length)
        
        # hidden: (n_layers * num_directions, batch_size, hidden_size)
        outputs, hidden = self.gru(pack)
        
        # unpack outputs: (max_length, batch_size, num_direction * hidden_size)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        
        # sum bidirectional outputs
        outputs = outputs[:, :, :self.hidden_dim] + outputs[:, :, self.hidden_dim:]
        return outputs, hidden


class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
    
    # current decoder state (ht): (1, batch_size, hidden_size) 
    # encoder output (hs): (max_length, batch_size, hidden_size)
    # return: (max_length, batch_size)
    def attn_dot(self, ht, hs):
        return torch.sum(ht * hs, dim=2) # sum across hidden_size
    
    # return: (batch_size, 1, max_length)
    def forward(self, ht, hs):
        # (batch_size, max_length)
        score = self.attn_dot(ht, hs).t()
        # normalize across each row so that the sum of each col in a row will be 1
        # and add one dimension
        return F.softmax(score, dim=1).unsqueeze(1)


class Decoder(nn.Module):
    # note: input_size=feature size, output_size = vocabulary size
    def __init__(self, method, embedding, hidden_size, output_size, n_layer=1, dropout=0):
        super(Decoder, self).__init__()
        self.method, self.hidden_size, self.output_size, self.n_layer = method, hidden_size, output_size, n_layer
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        # embed_size = hidden_size here
        self.gru = nn.GRU(hidden_size, hidden_size, n_layer, dropout=dropout)
        self.concat = nn.Linear(hidden_size*2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size) 
        self.attn = Attn(method, hidden_size)
    
    # decoder_input: (1, batch_size) one batch of words at a time
    # decoder_hidden: (n_layer * num_directional, batch_size, hidden_size)
    # encoder_output: (max_length, batch_size, hidden_size)
    def forward(self, decoder_input, decoder_hidden, encoder_output):
        # embeds: (1, batch_size, embed_size=hidden_size)
        embeds = self.embedding_dropout(self.embedding(decoder_input))

        # only one word, no need to pack padded sequence...
        
        # decoder_output: (1, batch_size, hidden_size)
        # hidden: (n_layers * num_directions, batch_size, hidden_size)
        decoder_output, hidden = self.gru(embeds, decoder_hidden)
        # calculate attention weight
        # score: (batch_size, 1, max_length)
        score = self.attn(decoder_output, encoder_output)
        # context: (batch_size, hidden_size)
        context = score.bmm(encoder_output.transpose(0, 1)).squeeze(1)
        # decoder_output: (batch_size, hidden_size)
        decoder_output = decoder_output.squeeze(0)
        # concat into (batch_size, hidden_size*2)
        linear_input = torch.cat((decoder_output, context), 1)
        concat_output = torch.tanh(self.concat(linear_input))
        # predict next word: (batch_size, vocab_size)
        output = F.softmax(self.out(concat_output), dim=1)
        return output, hidden


class InferenceDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(InferenceDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, input_t, lengths, max_target_len):
        encoder_output, encoder_hidden = encoder(input_t, lengths)
        decoder_input = torch.LongTensor([[SOS_TOKEN]])
        decoder_hidden = encoder_hidden[:decoder.n_layer]
        
        preds = torch.LongTensor()
        probs = torch.zeros([0])
        
        decoder_input = decoder_input.to(device)
        preds = preds.to(device)
        probs = probs.to(device)
        
        for _ in range(max_target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_output)
            prob, indx = torch.max(decoder_output, dim=1)
            # non teacher forcing
            decoder_input = torch.LongTensor([[indx]])
            preds = torch.cat((preds, indx), dim=0)
            probs = torch.cat((probs, prob), dim=0)
            if preds[-1] == EOS_TOKEN:
                break
        
        return preds, probs


