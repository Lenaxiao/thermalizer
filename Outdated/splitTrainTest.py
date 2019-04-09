import pandas as pd 
import numpy as np 

fname = 'seq_dic_nan.fasta'
trainingf = 'seq_dic_nan_train.txt'
testf = 'seq_dic_nan_test.txt'

def splitTraingTest(trainingf, testf, fname):
	training_set = open(trainingf, 'w')
	test_set = open(testf, 'w')

	content = []
	with open(fname, 'r') as f:
		for line in f :
			if not line.startswith('>'):
				content.append(line)

	print('original length: ', len(content))
	cnt_set = set(content)
	print('removed repeat: ', len(cnt_set))

	count = 0
	for line in cnt_set:
		count += 1
		if count <= 1000:
			test_set.write(line)
		elif count <= 101000:
			training_set.write(line)
		else:
			break

	content = []  # release storage
	training_set.close()
	test_set.close()


def makeObservations(fname, tfile):
	query = ['A', 'T', 'L', 'I']
	target = ['V', 'Q', 'E', 'R']

	temp = open(tfile, 'w');

	with open(fname, 'r') as f:
		for line in f :
			for ch in line:
				if ch in query:
					temp.write(target[query.index(ch)])
				else:
					temp.write(ch)
	temp.close()

# mapping character to index
# def map_character (dic, content):
# 	for i in range(len(content)):
# 		temp = list(content[i])
# 		for index, ch in enumerate(temp):
# 			temp[index] = dic[ch]
# 		content[i] = temp

# 	print('Replace characters of sequence with index in dictionary!')

# 	return content

# from keras.preprocessing import sequence 
# from keras.models import Sequential
# from keras.layers import Embedding, add, Dropout, Bidirectional, LSTM

# # perform skip_gram
# def skip_grams(content, dic, window_size):
# 	voc_size = len(dic)
# 	sampling_table = sequence.make_sampling_table(voc_size)
# 	couples = []
# 	labels = []
# 	for sec in content:
# 		couples_temp, labels_temp = sequence.skipgrams(sec, voc_size, window_size=window_size, sampling_table=sampling_table)
# 		couples.append(couples_temp)
# 		labels.append(labels_temp)
# 	print(len(content[0]), ' | ', len(couples[0]))
# 	return couples, labels



# # this is alternative to skip_grams to pad the sequence to same length
# # def padding_zeros():
# # 	pad_sequences()

# # hyperparameter: vec_dim, drop_val, nodes
# # input_dim: size of vocabulary (22 in this case)
# # output_dim: size of embedded vector (to be tuned)
# # input_length: length of input sequence (sentences need to be embedded into same dimension)
# def RNN_model(dic, vec_dim, x, max_length, drop_val, nodes):
# 	# three layers
# 	model = Sequential()
# 	model.add(Embedding(input_dim=len(dic), output_dim=vec_dim, input_length=max_length))
# 	model.add(Bidirectional(LSTM(nodes[0], return_sequences=True)))
# 	model.Dropout(drop_val)
# 	model.add(Bidirectional(LSTM(nodes[1], return_sequences=True)))
# 	model.Dropout(drop_val)
# 	model.add(Bidirectional(LSTM(nodes[2], return_sequences=False)))
# 	model.Dropout(drop_val)

# 	############### UNFINISHED ##############

# # splitTraingTest(trainingf, testf, fname)
# # makeObservations(trainingf, 'seq_dic_nan_train_y.txt')
# # makeObservations(testf, 'seq_dic_nan_test_y.txt')
# char_to_int, int_to_char, content = make_dictionary(trainingf)
# content = map_character(char_to_int, content)
# couples, labels = skip_grams(content, char_to_int, 3)

# use word2vec pakage
# def embeddingSeq(fname):
# 	content = []
# 	with open(fname, 'r') as f:
# 		content = f.readlines()

# 	for i in range(len(content)):
# 		content[i] = list(content[i])
	
# 	# train model
# 	# min_count is the minimum number of words to be accepted by the model
# 	model = Word2Vec(content, min_count=1)
# 	# summarize the loaded model
# 	print(model)
# 	# summarize vocabulary
# 	words = list(model.wv.vocab)
# 	print(words)
# 	# access vector for one word
# 	print(model['A'])
# 	# save model
# 	model.save('model.bin')
# 	# load model
# 	new_model = Word2Vec.load('model.bin')
# 	print(new_model)

