
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.contrib import layers, rnn
plt.switch_backend('agg')


# In[2]:


df = pd.read_csv("protein_seq_small.csv", index_col=0)
print(df.shape)
df.drop_duplicates(inplace=True)
print(df.shape)
input_text = df["meso_seq"].tolist()
output_text = df["thermal_seq"].tolist()


# In[3]:


# plot the sequence length
encode_lens = [len(seq) for seq in input_text]
decode_lens = [len(seq) for seq in output_text]
plt.subplots(figsize = (8, 5), dpi=500)
plt.hist(encode_lens, 200, normed=1, facecolor='blue', alpha=0.5, label="Mesophilic")
plt.hist(decode_lens, 200, normed=1, facecolor='red', alpha=0.5, label="Thermalphilic")
plt.xlim(0, 2000)
plt.xlabel('Sequence Length')
plt.ylabel('Probability')
plt.legend(loc='upper right')
plt.savefig("length.png", dpi=300)


# In[4]:


# chop off the sequence length
# with input seq length < 600 and output seq length < 500
input_index = [i for i, seq in enumerate(input_text) if len(seq) < 600]
output_index = [i for i, seq in enumerate(output_text) if len(seq) < 500]
index = set(input_index).intersection(output_index)
input_text = [input_text[i] for i in index]
output_text = [output_text[i] for i in index]


# In[5]:


print(len(input_index), "/", len(output_index))
print(len(input_text), "/", len(output_text))


# In[6]:


# make dictionarys for input and output
def make_dictionary (content):
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
    return char_to_int, int_to_char, word

###########
input_index, input_index_rev, input_char = make_dictionary(input_text)
output_index, output_index_rev, output_char = make_dictionary(output_text)


# In[7]:


print(input_index)


# In[8]:


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


# In[9]:


print("Source sequence is not in same length!")
print("1st: ", len(conv_input[0]))
print("2nd: ", len(conv_input[1]))


# In[10]:


encode_token = len(input_char)
decode_token = len(output_char)

print('Number of samples: ', len(input_text))
print('Number of unique input: ', encode_token)
print('Number of unique output: ', decode_token)
print('Max input/output length: ', max(len(i) for i in input_text), "/", max(len(i) for i in output_text))


# In[11]:


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


# In[12]:


# test
test_input = [[1, 2, 3], [4, 5, 6, 7]]
print(sequence_padding(test_input, 4))


# In[13]:


################## hyperparameters ###################
epochs = 200
batch_size = 65
num_layers = 2
num_units = 3
em_size = 64
l_rate = 0.001
att_size = 5
drop_val = 0
save_path="./model.ckpt"


# In[14]:


# add placeholders
inputs = tf.placeholder(tf.int32, [None, None], name='input')
targets = tf.placeholder(tf.int32, [None, None], name='target')
decode_length = tf.placeholder(tf.int32, shape=(batch_size), name="decode_length")
max_decode_len = tf.reduce_max(decode_length, name='max_dec_len')


# In[15]:


# setup bidirectional lstm
def get_a_lstm(num_units):
    cell = tf.nn.rnn_cell.LSTMCell(num_units)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1-drop_val)
    return cell


# In[16]:


# encoding layer
def encoding_layer(inputs, encode_token, em_size, num_layers, num_units, drop_val):
    
    # Maps a sequence of symbols to a sequence of embeddings.
    # embed_sequence is equivalant to:
    # encode_embed = tf.get_variable("encode_embedding",
    #                               initializer=tf.random_uniform([encode_token, em_size]),
    #                               dtype=tf.float32)
    # encode_embed_input = tf.nn.embedding_lookup(encode_embed, inputs)
    encode_embed_input = layers.embed_sequence(inputs, vocab_size=encode_token, embed_dim=em_size)

    stacked_lstm_fw = rnn.MultiRNNCell([get_a_lstm(num_units) for _ in range(num_layers)])
    stacked_lstm_bw = rnn.MultiRNNCell([get_a_lstm(num_units) for _ in range(num_layers)])
    outputs, final_states = tf.nn.bidirectional_dynamic_rnn(stacked_lstm_fw,
                                                            stacked_lstm_bw,
                                                            encode_embed_input,
                                                            dtype=tf.float32)
    output_fw, output_bw = outputs
    state_fw, state_bw = final_states
    encode_output = tf.concat([output_fw, output_bw], 2)
    encode_state = tf.concat([state_fw, state_bw], 2)
# ref 0: GPU setup
#     stacked_lstm = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers, num_units,
#                                                   direction='bidirectional',
#                                                   dropout=drop_val)
    return encode_output, encode_state


# In[17]:


# decoding layer (training + inference)
def decoding_layer(decode_input, output_index, em_size, num_units, num_layers, decode_length,
                   encode_output, encode_state, att_size, max_decode_len, drop_val, use_attention=True):
    
    # dynamically setup decoder embedding
    decode_embed = tf.get_variable("decode_embedding", 
                                   initializer=tf.random_uniform([len(output_index), em_size]),
                                   dtype=tf.float32)
    decode_cell = tf.contrib.rnn.MultiRNNCell([get_a_lstm(num_units) for _ in range(num_layers)])
    #decode_cell = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers, num_units, direction='bidirectional',
    #                                            dropout=drop_val)
    
    initial_state = encode_state
    if use_attention:
        attention_mech = tf.contrib.seq2seq.LuongAttention(num_units, encode_output)
        attention_cell = tf.contrib.seq2seq.AttentionWrapper(decode_cell,
                                                             attention_mech, 
                                                             attention_layer_size=att_size)
        initial_state = decode_cell.zero_state(dtype=tf.float32, batch_size=batch_size)

    # convert dense vector to the encoded one
    output_layer = tf.layers.Dense(len(output_index))
    
    train_output = decoding_train(decode_embed, decode_input, decode_length, decode_cell,
                                      initial_state, output_layer, max_decode_len)
        
    inference_output = decoding_inference(output_index, batch_size, decode_embed, decode_cell,
                                              initial_state, output_layer, max_decode_len)
    
    return train_output, inference_output


# In[18]:


# decode training part
def decoding_train(decode_embed, decode_input, decode_length, decode_cell, initial_state,
                   output_layer, max_decode_len):
    
    decode_embed_input = tf.nn.embedding_lookup(decode_embed, decode_input)
    
    # TrainingHelper is used in training process.
    # It feeds the ground truth at each step, pass embedded input.
    train_helper = tf.contrib.seq2seq.TrainingHelper(decode_embed_input,
                                                     decode_length)
    train_decoder = tf.contrib.seq2seq.BasicDecoder(decode_cell,
                                                    train_helper,
                                                    initial_state,
                                                    output_layer)
    train_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(train_decoder,
                                                            impute_finished=True,
                                                            maximum_iterations=max_decode_len)
    return train_outputs


# In[19]:


# decode inference part
def decoding_inference(output_index, batch_size, decode_embed, decode_cell, initial_state,
                       output_layer, max_decode_len):
    
    # GreedyEmbeddingHelper is used in inference process
    start_tokens = tf.tile(tf.constant([output_index['<s>']], dtype=tf.int32), [batch_size])
    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decode_embed, 
                                                                start_tokens, 
                                                                end_token=output_index['</s>'])
    inference_decoder = tf.contrib.seq2seq.BasicDecoder(decode_cell,
                                                        inference_helper,
                                                        initial_state,
                                                        output_layer)
    inference_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                impute_finished=True,
                                                                maximum_iterations=max_decode_len)
    return inference_outputs


# In[20]:


# wrapped model for sequences
def rnn_model(inputs, targets, encode_token, em_size, num_layers, num_units, output_index,
              batch_size, decode_length, att_size, max_encode_len, l_rate, drop_val):
    
    # encoding
    encode_output, encode_state = encoding_layer(inputs, encode_token, em_size, num_layers,
                                                 num_units, drop_val)
    # decoding
    train_output, inference_output = decoding_layer(targets, output_index, em_size,
                                                    num_units, num_layers, decode_length,
                                                    encode_output, encode_state, att_size,
                                                    max_encode_len, drop_val)
    # sample_id is the argmax of the RNN output logits.
    # logits = tf.matmul(inputs, weight) + bias
    train_logit = tf.identity(train_output.rnn_output, name='train_logit')
    train_preds = tf.identity(train_output.sample_id, name='train_pred')
    
    inference_preds = tf.identity(inference_output.sample_id, name='prediction')

    # compute train loss: weighted average cross-entropy (log-perplexity) per symbol
    # default: softmax_loss_function = sparse_softmax_cross_entropy_with_logits
    masks = tf.sequence_mask(decode_length, max_decode_len, dtype=tf.float32, name='mask')
    train_loss = tf.contrib.seq2seq.sequence_loss(train_logit, targets, masks, name='train_loss')
    
    # add optimizer
    training_opt = tf.train.AdamOptimizer(l_rate).minimize(train_loss)
    return train_loss, train_preds, inference_preds, training_opt


# In[21]:


# cut based on batch_size
def batch_div(data, batch_size):
    batched_data = []
    for i in range(len(data)//batch_size):
        batched_data.append(data[batch_size*i:batch_size*(i+1)])
    return batched_data


# In[22]:


def pad_length(input_batch, target_batch):
    batch_max_len = max([len(seq) for seq in input_batch])

    batch_pad_input = sequence_padding(input_batch, batch_max_len)
    batch_pad_target = sequence_padding(target_batch, batch_max_len, target=True)   

    decode_len_batch = []
    for target in batch_pad_target:
        decode_len_batch.append(len(target))
        
    return decode_len_batch


# In[23]:


## to be replaced by Bleu later
def get_accuracy(target, logits):
    return np.mean(np.equal(target, logits))


# In[24]:


test_target = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
test_logits = np.array([[1, 2, 1], [0, 4, 5], [3, 2, 8]])
get_accuracy(test_target, test_logits)


# In[25]:


########################### training process ############################
## early stopping tolerance: 25
def train(sess, train_input, val_input, train_target, val_target, encode_token, batch_size,
          epochs, train_loss, train_preds, inference_preds, training_opt, save_path):
    
    sess.run(tf.global_variables_initializer())
    
    all_loss = []
    all_val_acc = []
    all_train_acc = []
    # prepare for the validation set
    val_input_batches = batch_div(val_input, batch_size)
    val_target_batches = batch_div(val_target, batch_size)
    
    # prepare for the training
    input_batches = batch_div(train_input, batch_size)
    target_batches = batch_div(train_target, batch_size)
    
    for epoch in tqdm(range(epochs)):
        
        epoch_loss = 0
        train_accuracy = 0
        train_count = len(input_batches)
        
        for input_batch, target_batch in zip(input_batches, target_batches):
            decode_len_batch = pad_length(input_batch, target_batch)
            # training
            batch_loss, batch_pred, _ = sess.run([train_loss, train_preds, training_opt],
                                                  feed_dict={
                                                      inputs: input_batch,
                                                      targets: target_batch,
                                                      decode_length: decode_len_batch
                                                  })
            epoch_loss += batch_loss
            train_accuracy += get_accuracy(np.array(target_batch), np.array(batch_pred))
            train_accuracy /= train_count   
        all_train_acc.append(train_accuracy)
        all_loss.append(epoch_loss)
        
        # inference for training and validation set
        val_accuracy = 0
        val_count = len(val_input_batches)
        for val_input_batch, val_target_batch in zip(val_input_batches, val_target_batches):
            val_decode_len_batch = pad_length(val_input_batch, val_target_batch)
            val_batch_pred = sess.run(inference_preds,
                                      feed_dict={
                                          inputs: val_input_batch,
                                          decode_length: val_decode_len_batch
                                      })
            val_accuracy += get_accuracy(np.array(val_target_batch), np.array(val_batch_pred))
            val_accuracy /= val_count
        all_val_acc.append(val_accuracy)
        # display part
        if epoch != 0 and epoch % 50 == 0:
            print(f"Epoch: {epoch}, Train Loss: {all_loss[-1]}")
        
        
    saver = tf.train.Saver()
    saver.save(sess, save_path)
    print('Model Trained and Saved!')
    return all_loss, all_val_acc, all_train_acc


# In[26]:


# splits into train/test sets
train_input, val_input, train_target, val_target = train_test_split(conv_input,
                                                                    conv_output,
                                                                    test_size=0.1,
                                                                    random_state=7)
print("train samples: ", len(train_input))
print("validation samples: ", len(val_input))


# In[27]:


with tf.Session() as sess:
    train_loss, train_preds, inference_preds, training_opt = rnn_model(inputs, targets, encode_token,
                                                                       em_size, num_layers, num_units,
                                                                       output_index, batch_size, decode_length,
                                                                       att_size, max_decode_len, l_rate, drop_val)
    loss, val_acc, train_acc = train(sess, train_input, val_input,
                                     train_target, val_target, encode_token,
                                     batch_size, epochs, train_loss,
                                     train_preds, inference_preds,
                                     training_opt, save_path)
    
    plt.figure(figsize=(20, 10))
    plt.scatter(range(epochs), loss)
    plt.title('Learning Curve')
    plt.xlabel('Global step')
    plt.ylabel('Loss')
    plt.savefig("loss.png", pdi=300)


# In[28]:

plt.figure(figsize=(20, 10))
plt.scatter(range(epochs), train_acc, label='train accuracy')
plt.scatter(range(epochs), val_acc, label='validation accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Global step')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("acc.png", dpi=300)


# ### Improve:
# 0. <p style="text-decoration:line-through;">chop off sequence that larger than 500aa</p>
# 1. <p style="text-decoration:line-through;">bidirectional</p>
# 2. finish reference part
# 3. <p style="text-decoration:line-through;">training/validation set</p>
# 4. use BLEU to evaluate
# 5. GPU
