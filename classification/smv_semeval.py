import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, r2_score
from sklearn import preprocessing

import random
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from tensorflow.math import confusion_matrix
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Embedding, Input, LSTM, Dense, Bidirectional, Dropout, GRU
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.callbacks import EarlyStopping

from gensim.models import KeyedVectors


import matplotlib.pyplot as plt
import settings
import re

from collections import Counter
import statistics

from util import *



lstm_dim = 250
embedding_dim = 300
binary = True
batch_size = 1024
dim_arr = [10, 30, 50, 100, 200, 300]
arr_epochs = [100, 200, 300, 400, 500]
arr_activation_functions = ['tanh', 'relu', 'sigmoid', 'exponential']
arr_type_matrix_emb = ['vad']
embedding_type = ['glove', 'word2vec', 'numberbatch']

arr_pca = ['pca', 'nopca']
dir_datasets = settings.input_dir_emo_corpora + 'semeval/semeval_2013/'


class Sequencer():
	
	def __init__(self,
				 all_words,
				 max_words,
				 seq_len,
				 embedding_matrix
				):
		
		self.seq_len = seq_len
		self.embed_matrix = embedding_matrix
		"""
		temp_vocab = Vocab which has all the unique words
		self.vocab = Our last vocab which has only most used N words.
	
		"""
		temp_vocab = list(set(all_words))
		self.vocab = []
		self.word_cnts = {}
		"""
		Now we'll create a hash map (dict) which includes words and their occurencies
		"""
		for word in temp_vocab:
			# 0 does not have a meaning, you can add the word to the list
			# or something different.
			count = len([0 for w in all_words if w == word])
			self.word_cnts[word] = count
			counts = list(self.word_cnts.values())
			indexes = list(range(len(counts)))
		
		# Now we'll sort counts and while sorting them also will sort indexes.
		# We'll use those indexes to find most used N word.
		cnt = 0
		while cnt + 1 != len(counts):
			cnt = 0
			for i in range(len(counts)-1):
				if counts[i] < counts[i+1]:
					counts[i+1],counts[i] = counts[i],counts[i+1]
					indexes[i],indexes[i+1] = indexes[i+1],indexes[i]
				else:
					cnt += 1
		
		for ind in indexes[:max_words]:
			self.vocab.append(temp_vocab[ind])
					
	def textToVector(self,text):
		# First we need to split the text into its tokens and learn the length
		# If length is shorter than the max len we'll add some spaces (100D vectors which has only zero values)
		# If it's longer than the max len we'll trim from the end.
		tokens = text.split()
		print(tokens)
		len_v = len(tokens)-1 if len(tokens) < self.seq_len else self.seq_len-1
		print(len_v)
		vec = []
		for tok in tokens[:len_v]:
			try:
				print(tok)
				vec.append(self.embed_matrix[tok])
			except Exception as E:
				pass
		
		last_pieces = self.seq_len - len(vec)
		print(last_pieces)
		for i in range(last_pieces):
			vec.append(np.zeros(100,))

		vec = np.asarray(vec, dtype=object).flatten()
		print(np.shape(vec))
		
		return np.asarray(vec, dtype=object).flatten()
				
				
y_train, x_train, y_dev, x_dev, y_test, x_test = read_datasets(dir_datasets)

x = x_train
x.extend(x_dev)
x.extend(x_test)
print(len(x))

max_len_seq = max(len(s) for s in x)
path = settings.dir_embeddings_glove
word2vec = {}
for line in open(path):
	values = line.split()
	word2vec[str(values[0]).lower()] = np.asarray(values[1:], dtype='float32')

arr = []
for seq in x:
	for token in seq:
		arr.extend(token)
print('max_len_seq: ', max_len_seq)
sequencer = Sequencer(all_words = arr,
              max_words = 1200,
              seq_len = max_len_seq,
              embedding_matrix = word2vec
             )

test_vec = sequencer.textToVector("i am in love with you")
#print(test_vec)

exit()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train + x_dev)
x_train = tokenizer.texts_to_sequences(x_train)
x_dev = tokenizer.texts_to_sequences(x_dev)
x_test = tokenizer.texts_to_sequences(x_test)

# get the word to index mapping for input language
word2idx = tokenizer.word_index
print('Found %s unique input tokens.' % len(word2idx))

# determine maximum length input sequence
max_len_input = max(len(s) for s in x_train + x_dev)

x_train = pad_sequences(x_train, max_len_input, padding='pre', truncating='post')
x_dev = pad_sequences(x_dev, max_len_input, padding='pre', truncating='post')
x_test = pad_sequences(x_test, max_len_input, padding='pre', truncating='post')

##############################################################################################################3
word2vec = {}
#path = settings.dir_embeddings_glove #'/home/carolina/Documents/sota/Emotional-Embedding-master/counter_fitted_vectors-0.txresults/counter_fitted_vectors-0.txt'
#path = '/home/carolina/corpora/embeddings/emotions_embedings/sawe-tanh-pca-100-glove.txt'
emb_type = 'test_glove_vad_mms_dot_product_hstack_plus_bias_relu_pca'
#path = settings.dir_embeddings_word2vec 
#path = '/home/carolina/Documents/sota/counter-fitting-master/results/counter_fitted_vectors.txt'
path = '/home/carolina/embeddings/dense_model/emb/test_glove_vad_mms_dot_product_hstack_plus_bias_relu_pca.txt'
if emb_type != 'word2vec':
	for line in open(path):
		values = line.split()
		word2vec[str(values[0]).lower()] = np.asarray(values[1:], dtype='float32')
else:
	word2vec = KeyedVectors.load_word2vec_format(path, binary=True)
print(path)
count_missing_words = 0
# prepare embedding matrix
print('Filling pre-trained embeddings...')
num_words = len(word2idx) + 1
embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in word2idx.items():
	try:
		embedding_vector = word2vec[word]
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector
		else:
			embedding_matrix[i] = np.random.uniform(-0.25, 0.25, embedding_dim )
	except:
		embedding_matrix[i] = np.random.uniform(-0.25, 0.25, embedding_dim)

embedding_layer = Embedding(
		embedding_matrix.shape[0],
		embedding_matrix.shape[1],
		weights=[embedding_matrix],
		#input_length=max_len_input,
		trainable=False
)


input_ = Input(shape=(max_len_input,))
x = embedding_layer(input_)
bidirectional = GRU(150)#, recurrent_dropout=0.5))
x1 = bidirectional(x)
output = Dense(1, activation='sigmoid')(x1)#, kernel_regularizer=regularizers.l2(0.01))(x1)#, bias_regularizer=regularizers.l2(0.01))(x1)

arr_acc = []
arr_precision = []
arr_recall = []
arr_f1 = []

for run in range(1, 11):
	print('Run: ', run)
	model = Model(inputs=input_, outputs=output)
	model.compile('adam',#Adam(learning_rate=0.001),#'adam', 
		'binary_crossentropy', 
		metrics=['accuracy'])
	#model.summary()
	#exit()

	checkpoint_filepath = 'tmp/checkpoint_' + emb_type
	model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
			filepath=checkpoint_filepath,
			save_weights_only=True,
			monitor='val_accuracy',
			mode='max',
			save_best_only=True)


	early_stop = EarlyStopping(monitor='val_accuracy', patience=10)

	r = model.fit(x_train, y_train, validation_data=(x_dev, y_dev), 
		batch_size=512, epochs=50, verbose=0, callbacks=[model_checkpoint_callback, early_stop])


	# The model weights (that are considered the best) are loaded into the
	# model.
	model.load_weights(checkpoint_filepath)

	pred = model.predict(x_test, verbose=1)
	pred = np.where(pred > 0.5, 1, 0)

	precision = precision_score(y_true=y_test, y_pred=pred, labels=[0, 1], pos_label=1, average='binary')
	recall = recall_score(y_true=y_test, y_pred=pred, labels=[0, 1], pos_label=1, average='binary')
	f1 = f1_score(y_true=y_test, y_pred=pred, labels=[0, 1], pos_label=1, average='binary')
	acc = accuracy_score(y_true=y_test, y_pred=pred)
	r2 = r2_score(y_true=y_test, y_pred=pred)


	#print('Lexico: ', lexico)
	lstm_dim_vec = 300
	print('Emo_emb_size: ', lstm_dim_vec)
	print('acc: ', acc)
	print('precision: ', precision)
	print('recall: ', recall)
	print('f1: ', f1)
	exit()
	arr_acc.append(acc)
	arr_precision.append(precision)
	arr_recall.append(recall)
	arr_f1.append(f1)

	# loss
	'''plt.plot(r.history['loss'], label='loss')
	plt.plot(r.history['val_loss'], label='val_loss')
	plt.legend()
	plt.show()

	# accuracies
	plt.plot(r.history['accuracy'], label='acc')
	plt.plot(r.history['val_accuracy'], label='val_acc')
	plt.legend()
	plt.show()
	'''

	cf_matrix = confusion_matrix(labels=y_test, predictions=pred, num_classes=2)
	cf_matrix = np.array(cf_matrix)
	rows, columns = np.shape(cf_matrix)
	path_cf = 'confusion_matrix'
	print(cf_matrix)
	if not os.path.exists(path_cf):
		with open(path_cf, 'w') as file:
			file.close()
	with open(path_cf + '/' + emb_type + '.txt', 'a') as file:
		file.write('run: ' + str(run) + '\n')
		for x in range(rows):
			for y in range(columns):
				file.write(str(cf_matrix[x][y]) + (' ' if y < rows-1 else '\n'))
		file.write('\n')
		file.close()
	#fig, ax = plt.subplots(figsize=(15,10)) 
	#sn.heatmap(cf_matrix, linewidths=1, annot=True, ax=ax, fmt='g')
	#plt.show()

	file_results = 'results_semeval_in_deep.csv'
	if not os.path.exists(file_results):
		with open(file_results, 'w') as file:
			file.write('run\tembeddings\taccuracy\tprecision\trecall\tf1_score\n')
			file.close()

	#str_dir = 'sent_emb_' + emb_type + '_' + str(embedding_dimention) + '_' + act + 
	#		'_e'+ str(epoch) + '_' + apply_pca +'_' + type_matrix_emb + '.txt'

	with open(file_results, 'a') as file:
		file.write(str(run) + '\t' + emb_type + '\t%.6f\t%.6f\t%.6f\t%.6f\n' % (acc, precision, recall, f1))
		file.close()


file_results = 'results_semeval.csv'
if not os.path.exists(file_results):
	with open(file_results, 'w') as file:
		file.write('embeddings\taccuracy\tprecision\trecall\tf1_score\n')
		file.close()

with open(file_results, 'a') as file:
	file.write(emb_type + '\t%.6f (%.4f)\t%.6f (%.4f)\t%.6f (%.4f)\t%.6f (%.4f)\n' % (statistics.mean(arr_acc), 
		statistics.pstdev(arr_acc), statistics.mean(arr_precision), statistics.pstdev(arr_precision), 
		statistics.mean(arr_recall), statistics.pstdev(arr_recall), statistics.mean(arr_f1), statistics.pstdev(arr_f1)))
	file.close()
