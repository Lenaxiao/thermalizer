from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np

# vectorize data
input_text = []
output_text = []
input_character = set()
output_character = set()

def make_dictionary (fname):
    content = []
    with open(fname, 'r') as f:
        content = f.readlines()

    word = set()
    for line in content:
        for ch in line:
            word.add(ch)
    word = sorted(list(word))
    print('Length of unique words in dictionary: ', len(word))

    # make index table
    char_to_int = {w: i for i, w in enumerate(word)}
    int_to_char = {v: k for k, v in char_to_int.items()}
    print('Lookup table 1 (map chars to integer): ', char_to_int)
    print('Lookup table 2 (map integer to chars: ', int_to_char)

    return char_to_int, int_to_char, content, word

input_index, input_index_rev, input_text, input_char = make_dictionary('seq_dic_nan_test.txt')
output_index, output_index_rev, output_text, output_char = make_dictionary('seq_dic_nan_test_y.txt')

encode_token = len(input_char)
decode_token = len(output_char)
max_encode_len = max([len(txt) for txt in input_text])
max_decode_len = max([len(txt) for txt in output_text])

print('Number of samples: ', len(input_text))
print('Number of unique input: ', encode_token)
print('Number of unique output: ', decode_token)
print('Max input length: ', max_encode_len)
print('Max output length: ', max_decode_len)

# 3d matrix
# input --> encode --> decode --> node --> decode 
encode_input = np.zeros(
    (len(input_text), max_encode_len, encode_token), dtype='float32')
decode_input = np.zeros(
    (len(input_text), max_decode_len, decode_token), dtype='float32')
decode_output = np.zeros(
    (len(output_text), max_decode_len, decode_token), dtype='float32')
print('Dimension(words_number * padded_word * features)')
print('encode input', encode_input.shape)
print('decode input: ', decode_input.shape)
print('decode output: ', decode_output.shape)

for i, (input_line, output_line) in enumerate(zip(input_text, output_text)):
    for t, char in enumerate(input_line):
        encode_input[i, t, input_index[char]] = 1
    for t, char in enumerate(output_line):
        decode_input[i, t, output_index[char]] = 1
        if (t > 0):
            decode_output[i, t - 1, output_index[char]] = 1

# define parameters that can be tuned
batch_size = 64
epochs = 100
latent_dim = 256

# input --> LSTM(encoder) -->output(input) --> LSTM(decoder) --> output
# still vector to vector
encoder_inputs = Input(shape=(None, encode_token))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

#set up decoder layer
decoder_inputs = Input(shape=(None, decode_token))
decoder = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder(decoder_inputs, initial_state=encoder_states)

decoder_dense = Dense(decode_token, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encode_input, decode_input], decode_output, batch_size=batch_size, epochs=epochs,
         validation_split=0.2)
model.save('example.h5')