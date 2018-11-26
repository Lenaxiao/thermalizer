
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm


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
    seq_len = []
    for i in range(len(content)):
        seq_len.append(len(content[i]))
        temp = list(content[i])
        for index, ch in enumerate(temp):
            temp[index] = dic[ch]
        content[i] = temp
        
    return content, seq_len

###########
conv_input, encode_seq_len = map_character(input_index, input_text)
conv_output, decode_seq_len = map_character(output_index, output_text)


# In[5]:


print("Source sequence is not in same length!")
print("1st: ", len(conv_input[0]))
print("2nd: ", len(conv_input[1]))


# In[6]:


# plot the sequence length
## considering cutoff later
get_ipython().run_line_magic('matplotlib', 'inline')
encode_lens = [len(seq) for seq in conv_input]
plt.subplots(figsize = (8, 5), dpi = 600)
n, bins, patches = plt.hist(encode_lens, 30, normed=1, facecolor='blue', alpha=0.5)
plt.xlabel('Sequence Length')
plt.ylabel('Probability')
plt.show()


# In[7]:


encode_token = len(input_char)
decode_token = len(output_char)

print('Number of samples: ', len(input_text))
print('Number of unique input: ', encode_token)
print('Number of unique output: ', decode_token)
print('Max input/output length: ', max([len(txt) for txt in input_text]))


# In[8]:


# in this dataset, input and output pair always have the same length.
# pad short sequence to the longest one
# index of <PAD> is zero.
def sequence_padding(content, max_len, target=False):
    head_index = output_index['<s>']
    pad_index = input_index['<PAD>']
    for index, seq in enumerate(content):
        pad_to_seq = [pad_index]*(max_len - len(seq))
        if target:
            content[index] = [head_index] + [*seq, *pad_to_seq]
        else:
            content[index] = [*seq, *pad_to_seq]
        
    return content


# In[9]:


# test
test_input = [[1, 2, 3], [4, 5, 6, 7]]
print(sequence_padding(test_input, 4))


# In[10]:


################## hyperparameters ###################
epochs = 800
batch_size = 128
rnn_size = 50 # number of LSTM cells
num_layers = 2
num_units = 3
em_size = 60
l_rate = 0.001
att_size = 5
#drop_val = 0.5
save_path="/model_history/model.ckpt"


# In[11]:


# cut based on batch_size
def batch_div(data, batch_size):
    batched_data = []
    for i in range(len(data)//batch_size):
        batched_data.append(data[batch_size*i:batch_size*(i+1)])
    return batched_data


# In[12]:


# add placeholders
inputs = tf.placeholder(tf.int32, [None, None], name='input')
targets = tf.placeholder(tf.int32, [None, None], name='target')
decode_length = tf.placeholder(tf.int32, shape=(batch_size), name="decode_length")
max_decode_len = tf.reduce_max(decode_length, name='max_dec_len')
# lr = tf.placeholder(tf.float32)


# In[13]:


############################## Build Model ################################
# encoding layer
def encoding_layer(inputs, encode_token, em_size, num_layers, num_units):
    
    # encoded input embedding
    #input_embed = tf.contrib.layers.embed_sequence(inputs, vocab_size=encode_token,
    #                                              embed_dim=em_size)
    encode_embed = tf.get_variable("encode_embedding",
                                   initializer=tf.random_uniform([encode_token, em_size]),
                                   dtype=tf.float32)
    encode_embed_input = tf.nn.embedding_lookup(encode_embed, inputs)
    ## setup drop out later
    ## setup bidirectional later
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units)
    #lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(num_units)
    def get_a_lstm(num_units):
        return tf.nn.rnn_cell.BasicLSTMCell(num_units)
    stacked_lstm = tf.contrib.rnn.MultiRNNCell([get_a_lstm(num_units) for _ in range(num_layers)])
    #stacked_lstm_bw = tf.contrib.rnn.MultiRNNCell([lstm_cell_bw] * num_layers)
    
    encode_output, encode_state = tf.nn.dynamic_rnn(stacked_lstm,
                                                    encode_embed_input,
                                                    dtype=tf.float32)
    return encode_output, encode_state


# In[14]:


# decoding layer (training + inference)
def decoding_layer(decode_input, output_index, em_size, num_units, num_layers,
                   decode_length, encode_output, att_size, max_decode_len):
    
    # decoder input embedding
    decode_embed = tf.get_variable("decode_embedding", [len(output_index), em_size])
    decode_embed_input = tf.nn.embedding_lookup(decode_embed, decode_input)
    
    def get_a_lstm(num_units):
        return tf.nn.rnn_cell.BasicLSTMCell(num_units)
    decode_cell = tf.contrib.rnn.MultiRNNCell([get_a_lstm(num_units) for _ in range(num_layers)])
    
    # TrainingHelper feeds the ground truth at each step, pass embedded input.
    train_helper = tf.contrib.seq2seq.TrainingHelper(decode_embed_input, decode_length)
    
    # GreedyEmbeddingHelper is used in inference process
    start_tokens = tf.tile(tf.constant([output_index['<s>']], dtype=tf.int32), [batch_size])
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
                attention_cell = tf.contrib.seq2seq.AttentionWrapper(decode_cell,
                                                                     attention_mech, 
                                                                     attention_layer_size=att_size)
                initial_state = decode_cell.zero_state(dtype=tf.float32, batch_size=batch_size)
            else:
                initial_state = encode_state

            # convert dense vector to the encoded one
            ## Dense or OutputProjectionWrapper ???
            output_layer = tf.layers.Dense(len(output_index))
            decoder = tf.contrib.seq2seq.BasicDecoder(decode_cell, helper, initial_state,
                                                      output_layer)
            outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                        impute_finished=True,
                                                                        maximum_iterations=max_decode_len)
            return outputs
    
    train_output = decode(train_helper)
    inference_output = decode(inference_helper, reuse=True)
    
    return train_output, inference_output


# In[15]:


# wrapped model for sequences
def rnn_model(inputs, targets, encode_token, em_size, num_layers, num_units, output_index,
              batch_size, decode_length, att_size, max_encode_len, l_rate):
    
    # encoding
    encode_output, encode_state = encoding_layer(inputs, encode_token,
                                                 em_size, num_layers, num_units)
    # decoding
    train_output, inference_output = decoding_layer(targets, output_index, em_size,
                                                    num_units, num_layers, decode_length,
                                                    encode_output, att_size, max_encode_len)
    # rnn_output is the output of the decoding cell
    train_logit = tf.identity(train_output.rnn_output, name='logit')
    inference_logit = tf.identity(inference_output.sample_id, name='prediction')

    # compute loss
    masks = tf.sequence_mask(decode_length, max_decode_len, dtype=tf.float32, name='mask')
    train_loss = tf.contrib.seq2seq.sequence_loss(train_logit, targets, masks, name='loss')

    # sample_id is the argmax of the rnn output
    valid_predictions = tf.identity(train_output.sample_id, name='valid_pred')
    predictions = tf.identity(inference_output.sample_id, name='prediction')

    # add optimizer
    training_opt = tf.train.RMSPropOptimizer(l_rate).minimize(train_loss)
    return train_loss, training_opt


# In[16]:


########################### training process ############################
def train(sess, conv_input, conv_output, batch_size, epochs, train_loss, training_opt):
    
    sess.run(tf.global_variables_initializer())
    
    all_loss = []

    input_batches = batch_div(conv_input, batch_size)
    target_batches = batch_div(conv_output, batch_size)
    for epoch in tqdm(range(epochs)): 
        epoch_loss = 0
        for input_batch, target_batch in zip(input_batches, target_batches):

            batch_max_len = max([len(seq) for seq in input_batch])

            batch_pad_input = sequence_padding(input_batch, batch_max_len)
            batch_pad_target = sequence_padding(target_batch, batch_max_len, target=True)   

            decode_len_batch = []
            for target in batch_pad_target:
                decode_len_batch.append(len(target))

            batch_loss, _ = sess.run([train_loss, training_opt],
                                     feed_dict={
                                         inputs: input_batch,
                                         targets: target_batch,
                                         decode_length: decode_len_batch,
                                     })
            epoch_loss += batch_loss
            
        all_loss.append(epoch_loss)
        ## early stopping tolerance: 25
        
        if epoch != 0 and epoch % 10 == 0:
            print(f"Epoch: {epoch}, Loss: {all_loss[-1]}")
        
        
    saver = tf.train.Saver()
    saver.save(sess, save_path)
    print('Model Trained and Saved!')
    return all_loss


# In[17]:


# def inference(sess, data, load_ckpt):
#     restorer = tf.train.Saver()
#     restorer.restore(sess, save_path)
#     print('Model Restored!')


# In[18]:


with tf.Session() as sess:
    train_loss, training_opt = rnn_model(inputs, targets, encode_token, em_size,
                                         num_layers, num_units, output_index,
                                         batch_size, decode_length, att_size,
                                         max_decode_len, l_rate)
    loss = train(sess, conv_input, conv_output, batch_size, epochs, train_loss, training_opt)


# In[ ]:


# def cross_validation(sess, conv_input, conv_output):
#     result = []
#     cv_loss = []
#     stratk = KFold(n_splits=5, random_state=7)
#     splits = stratk.split(conv_input, conv_output)
#     for train_index, val_index in splits:
#         train_input = conv_input[train_index]
#         train_target = conv_output[train_index]
#         val_input = conv_input[val_index]
#         val_target = conv_output[val_index]
#         train_loss, training_opt, predictions = rnn_model(inputs, targets, encode_token,
#                                                           em_size, num_layers, num_units,
#                                                           output_index, batch_size,
#                                                           decode_length, att_size,
#                                                           max_decode_len, l_rate)
#         loss = train(sess, conv_input, conv_output, batch_size, epochs, train_loss,
#                      training_opt, predictions)
#         cv_loss.append(loss)
#         result.append(sess.run(predictions, feed_dict={x: val_input, y: val_target}))


# In[ ]:


for i in range(10):
    if i != 0 and i % 2 == 10:
        print(f"Epoch: {epoch}, Loss: {all_loss[-1]}")

