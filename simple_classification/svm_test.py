import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, r2_score
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.svm import SVC
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
import re

from collections import Counter
import statistics

from util import *


dir_datasets = '/home/carolina/corpora/emotion_datasets/semeval/semeval_2017/'



def get_embeddings(input_arr, word2vec, idx2word):
	idx = 0
	embedding_matrix = np.zeros((len(input_arr), 300))
	for sent in input_arr:
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


tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train + x_dev)
x_train = tokenizer.texts_to_sequences(x_train)
x_dev = tokenizer.texts_to_sequences(x_dev)
x_test = tokenizer.texts_to_sequences(x_test)

# get the word to index mapping for input language
word2idx = tokenizer.word_index
idx2word = {y: x for x, y in word2idx.items()}

print('Found %s unique input tokens.' % len(word2idx))


##############################################################################################################3
word2vec = {}
path = '/home/carolina/corpora/embeddings/glove/glove.42B.300d.txt'
emb_type = 'glove'
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
start_time = time.time()
x_train = get_embeddings(x_train, word2vec, idx2word)
x_dev = get_embeddings(x_dev, word2vec, idx2word)
x_test = get_embeddings(x_test, word2vec, idx2word)
print("Time filling embeddings: ", (time.time() - start_time))


print('Starting classification...')
start_time = time.time()
svm_classifier = SVC(kernel='poly', C=1, degree=3)
svm_classifier.fit(x_train,y_train)
print('Training ended')
print("Time training: ", (time.time() - start_time))

print('-----------------------------------------------------------------')
start_time = time.time()
pred_dev = svm_classifier.predict(x_dev)
dev_accuracy = accuracy_score(y_dev, pred_dev)
dev_f1 = f1_score(y_dev, pred_dev, average='weighted')
print('Accuracy (dev): ', "%.2f" % (dev_accuracy*100))
print('F1 (dev): ', "%.2f" % (dev_f1*100))
print("Time predicting dev: ", (time.time() - start_time))


print('-----------------------------------------------------------------')
start_time = time.time()
pred_test = svm_classifier.predict(x_test)
test_accuracy = accuracy_score(y_test, pred_test)
test_f1 = f1_score(y_test, pred_test, average='weighted')
print('Accuracy (test): ', "%.2f" % (test_accuracy*100))
print('F1 (test): ', "%.2f" % (test_f1*100))
print("Time predicting test: ", (time.time() - start_time))


exit()

#print('Lexico: ', lexico)
lstm_dim_vec = 300
print('Emo_emb_size: ', lstm_dim_vec)
print('acc: ', acc)
print('precision: ', precision)
print('recall: ', recall)
print('f1: ', f1)
arr_acc.append(acc)
arr_precision.append(precision)
arr_recall.append(recall)
arr_f1.append(f1)

cf_matrix = multilabel_confusion_matrix(y_true=y_test, y_pred=pred)
cf_matrix = np.array(cf_matrix)
#rows, columns = np.shape(cf_matrix)
#path_cf = 'confusion_matrix'
print(cf_matrix)


# loss
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# accuracies
plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()

exit()
