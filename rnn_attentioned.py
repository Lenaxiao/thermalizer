
# coding: utf-8

# In[1]:


# from keras.models import Model
# from keras.layers import Input, LSTM, Dense
import numpy as np
import pandas as pd
import tensorflow as tf


# In[2]:


# make dictionarys for input and output
def make_dictionary (fname):
    content = []
    with open(fname, 'r') as f:
        content = [line.rstrip() for line in f]

    word = set()
    for line in content:
        for ch in line:
            word.add(ch)
    word = sorted(list(word))
    print('Length of unique words in dictionary: ', len(word))
    char_to_int = {'<PAD>': 0, '<UNK>': 1, '<s>': 2, '</s>': 3}
    init_n = len(char_to_int)
    # make index table
    for i, w in enumerate(word, init_n):
        char_to_int[w] = i
    int_to_char = {v: k for k, v in char_to_int.items()}
    return char_to_int, int_to_char, content, word

###########
input_index, input_index_rev, input_text, input_char = make_dictionary('seq1.txt')
output_index, output_index_rev, output_text, output_char = make_dictionary('seq2.txt')


# In[3]:


print(input_index)


# In[4]:


# mapping character to index for all sentences
def map_character (dic, content):
    for i in range(len(content)):
        temp = list(content[i])
        for index, ch in enumerate(temp):
            temp[index] = dic[ch]
        content[i] = temp
        
    return content

###########
conv_input = map_character(input_index, input_text)
conv_output = map_character(output_index, output_text)


# In[5]:


print("Source sequence is not in same length!")
print("1st: ", len(conv_input[0]))
print("2nd: ", len(conv_input[1]))


# In[6]:


encode_token = len(input_char)
decode_token = len(output_char)
max_encode_len = max([len(txt) for txt in input_text])
max_decode_len = max_encode_len

print('Number of samples: ', len(input_text))
print('Number of unique input: ', encode_token)
print('Number of unique output: ', decode_token)
print('Max input length: ', max_encode_len)
print('Max output length: ', max_decode_len)


# In[7]:


# in this dataset, input and output pair always have the same length.
# pad short sequence to the longest one
# index of <PAD> is zero.
def sequence_padding(content, max_len, pad_index):
    for index, seq in enumerate(content):
        pad_to_seq = [pad_index]*(max_len - len(seq))
        content[index] = [*seq, *pad_to_seq]
    return content

pad_index = input_index['<PAD>']
padded_input = sequence_padding(conv_input, max_encode_len, pad_index)
padded_target = sequence_padding(conv_output, max_decode_len, pad_index)


# In[8]:


min_len = min([len(seq) for seq in padded_input])
if min_len == max_encode_len:
    print("All sequence padded to the same length!")
else:
    print("Sequence padding fails!")


# In[9]:


################## hyperparameters ###################
epochs = 0
batch_size = 128
rnn_size = 50 # number of LSTM cells
num_layers = 2
num_units = 3
em_size = 60
l_rate = 0.001
att_size = 5
#drop_val = 0.5


# In[10]:


# add placeholders
inputs = tf.placeholder(tf.int32, [None, None], name='input')
targets = tf.placeholder(tf.int32, [None, None], name='target')
decoder_length = tf.placeholder(tf.int32, shape=(batch_size), name="decoder_length")
# lr = tf.placeholder(tf.float32)


# In[11]:


# encoding layer
def encoding_layer(inputs, encode_token, em_size, num_layers, num_units):
    
    # encoded input embedding
    input_embed = tf.contrb.layers.embed_sequence(inputs, vocab_size=encode_token, embed_dim=em_size)
    
    ###### setup drop out later ######
    def lstm_cell():
        return tf.contrib.rnn.BasicLSTMCell(num_units)
    stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(num_layers)])
    encode_output, encode_state = tf.nn.dynamic_rnn(stacked_lstm, input_embed, dtype=tf.float32)
    return encode_output, encode_state


# In[16]:


# decoder input process: add <s> to the start
# strided_slice simply cut data into chunks with number of sequence equals to batch_size
def process_decode_input(output_index, targets, batch_size):
    head_index = output_index['<s>']
    ending = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1, 1])
    decode_input = tf.concat([tf.fill([batch_size, 1], head_index), ending], 1)
    
    return decode_input


# In[23]:


##### test for decoder input #####
sess = tf.InteractiveSession()
test_targets = np.reshape(np.arange(batch_size * max_encode_len, dtype=np.int32), (batch_size, max_encode_len))
test_decode_input = process_decode_input(output_index, test_targets, batch_size)
print(sess.run(test_decode_input, {targets: test_targets})[:2])
sess.close()


# In[37]:


# decoding layer (training + inference)
def decoding_layer(decode_input, output_index, em_size, num_units, num_layers,
                   decoder_length, encode_output, att_size, max_encode_len):
    
    # decoder input embedding
    decode_embed = tf.get_variable("decode_embedding", [len(output_index), em_size])
    decode_embed_input = tf.nn.embedding_lookup(decode_embed, decode_input)
    
    def lstm_cell():
        return tf.contrib.rnn.BasicLSTMCell(lstm_size)
    decode_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(num_layers)])
    
    # TrainingHelper feeds the ground truth at each step, pass embedded input.
    train_helper = tf.contrib.seq2seq.TrainingHelper(decode_embed_input, decoder_length)
    
    # GreedyEmbeddingHelper is used in inference process
    start_tokens = tf.tile(tf.constant(output_index['<s>'], dtype=tf.int32), [batch_size])
    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decode_embed, 
                                                                start_tokens, 
                                                                end_token=output_index['</s>'])
    
    # training and inference share the same structure except for the helper function
    def decode(helper, use_attention=True, reuse=None):
        with tf.variable_scope("decode", reuse=reuse):
            if use_attention:
                # LuongAttention accept: num_units, memory
                attention_mech = tf.contrib.seq2seq.LuongAttention(num_units, encode_output)
                # AttentionWrapper accept: cell, attention_mechanism, attention_layer_size
                attention_cell = tf.contrib.seq2seq.AttentionWrapper(decode_cell, attention_mech, 
                                                                    attention_layer_size=att_size)
                initial_state = decode_cell.zero_state(dtype=tf.float32, batch_size=batch_size)
            else:
                initial_state = encode_state

            # convert dense vector to the encoded one
            # Dense or OutputProjectionWrapper ???
            output_layer = tf.layers.Dense(len(output_index))
            decoder = tf.contrib.seq2seq.BasicDecoder(decode_cell, helper, initial_state,
                                                      output_layer)
            outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                        impute_finished=True,
                                                                        maximum_iterations=max_encode_len)
            return outputs
    
    train_output = decode(train_helper)
    inference_output = decode(inference_helper, reuse=True)
    
    return train_output, inference_output


# In[38]:


# wrapped model for sequences
def rnn_model(inputs, targets, encode_token, em_size, num_layers, num_units, output_index,
          batch_size, decoder_length, att_size, max_encode_len):
    
    # encoding
    encode_output, encode_state = encoding_layer(inputs, encode_token,
                                                 em_size, num_layers, num_units)
    # add start token to decoding inputs
    decode_input = process_decode_input(output_index, targets, batch_size)
    
    # decoding
    train_output, inference_output = decoding_layer(decode_input, output_index, em_size,
                                                    num_units, num_layers, decoder_length,
                                                    encode_output, att_size, max_encode_len)
    return train_output, inference_output

