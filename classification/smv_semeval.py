import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, r2_score
from sklearn import preprocessing
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.model_selection import cross_val_score, GridSearchCV

import time
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
dir_datasets = settings.input_dir_emo_corpora + 'semeval/semeval_2017/'



def get_embeddings(input_arr, word2vec, idx2word):
	idx = 0
	embedding_matrix = np.zeros((len(input_arr), 300))
	for sent in input_arr:
		#print(sent)
		counter = 0
		arr_emb = np.zeros(300)
		for word_index in sent:
			vec = word2vec.get(idx2word[word_index])
			arr_emb = arr_emb + (vec if vec is not None else np.random.uniform(-0.25, 0.25, 300))
			counter += 1
		embedding_matrix[idx] = arr_emb / counter
		idx += 1

	return embedding_matrix



y_train, x_train, y_dev, x_dev, y_test, x_test, classes = read_datasets(dir_datasets)

'''x = x_train
x.extend(x_dev)	
y = np.concatenate((y_train, y_dev), axis=0)

x_train = x
y_train = y'''


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

x_train = pad_sequences(x_train, max_len_input, padding='post', truncating='post')
x_dev = pad_sequences(x_dev, max_len_input, padding='post', truncating='post')
x_test = pad_sequences(x_test, max_len_input, padding='post', truncating='post')

##############################################################################################################3
word2vec = {}
print('Loading embeddings...')
path = settings.dir_embeddings_glove
emb_type = 'glove' 
#emb_type = 'combined_class_reg'
#path = '/home/carolina/embeddings/dense_model/emb/last_version' + emb_type + '.txt'
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

gamma_arr = [0.01, 0.1, 1, 10]
c_arr = [0.1, 1, 10, 100]
#for gamma in gamma_arr:
#	for c in c_arr:
print('-----------------------------------------------------------------')
print('kernel: rbf')
print('gamma: ', gamma)
print('c: ', c)
svm_classifier = SVC(kernel='poly', gamma=0.0001, C=0.1)
svm_classifier.fit(x_train,y_train)
pred_dev = svm_classifier.predict(x_dev)
dev_accuracy = accuracy_score(y_dev, pred_dev)
dev_f1 = f1_score(y_dev, pred_dev, average='micro')
print('---------')
print('Accuracy (dev): ', "%.2f" % (dev_accuracy*100))
print('F1 (dev): ', "%.2f" % (dev_f1*100))
print('---------')
pred_test = svm_classifier.predict(x_test)
test_accuracy = accuracy_score(y_test, pred_test)
test_f1 = f1_score(y_test, pred_test, average='micro')
print('Accuracy (test): ', "%.2f" % (test_accuracy*100))
print('F1 (test): ', "%.2f" % (test_f1*100))

with open('results_svm.csv', 'a') as file:
	file.write(str(gamma) + '|' + str(c) + '|%.4f|%.4f\n'% (dev_f1, test_f1))
	file.close()


'''
print('Starting classification...')
start_time = time.time()
svm_classifier = SVC(kernel='rbf', gamma=0.0001, C=0.1)
#svm_classifier = NuSVC(nu=0.03)
svm_classifier.fit(x_train,y_train)
print('Training ended')
print("Time training: ", (time.time() - start_time))


print('-----------------------------------------------------------------')
start_time = time.time()
pred_dev = svm_classifier.predict(x_dev)
dev_accuracy = accuracy_score(y_dev, pred_dev)
dev_f1 = f1_score(y_dev, pred_dev, average='micro')
print('Accuracy (dev): ', "%.2f" % (dev_accuracy*100))
print('F1 (dev): ', "%.2f" % (dev_f1*100))
print("Time predicting dev: ", (time.time() - start_time))

print('-----------------------------------------------------------------')
start_time = time.time()
pred_test = svm_classifier.predict(x_test)
test_accuracy = accuracy_score(y_test, pred_test)
test_f1 = f1_score(y_test, pred_test, average='micro')
print('Accuracy (test): ', "%.2f" % (test_accuracy*100))
print('F1 (test): ', "%.2f" % (test_f1*100))
print("Time predicting test: ", (time.time() - start_time))

'''


