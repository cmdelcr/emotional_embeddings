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

from nltk.corpus import stopwords


import matplotlib.pyplot as plt
import re
import string
punctuation = string.punctuation
stop_words_list = stopwords.words('english')

from collections import Counter
import statistics

from util import *


#dir_datasets = '/home/carolina/corpora/emotion_datasets/semeval/semeval_2017/'
dir_datasets = '/home/carolina/corpora/emotion_datasets/'



def get_embeddings(input_arr, word2vec, idx2word, label):
	idx = 0
	embedding_matrix = np.zeros((len(input_arr), 303))
	for sent in input_arr:
		counter = 0
		arr_emb = np.zeros(303)
		for word_index in sent:
			vec = word2vec.get(idx2word[word_index])
			arr_emb = arr_emb + (vec if vec is not None else np.random.uniform(-0.25, 0.25, 303))
			counter = counter + 1
		
		embedding_matrix[idx] = arr_emb / counter
			
		idx += 1

	return embedding_matrix

def preprocessing(sent):
	sent = re.sub(r'\n', '', re.sub(r'á', '', sent))
	sent = re.sub(r'\s+', ' ', sent.strip())
	sent = re.sub(r'[' + punctuation + ']+', '', sent)
	sent = re.sub(r'[0-9]+', '', sent)

	sent_aux = ''
	for token in sent.split():
		if token not in stop_words_list:
			sent_aux += token + ' '

	sent = sent_aux.strip()
	#print(sent)

	return sent

name_file = 'DATA' # isear, DATA
if name_file == 'isear':
	df = pd.read_csv(settings.input_dir_emo_corpora + 'isear/' + name_file + '.csv', delimiter=',', header=None)
else:
	df = pd.read_csv(settings.input_dir_emo_corpora + 'isear/' + name_file + '.csv', delimiter=',')

# Remove 'No response' row value in isear.csv
#df['Field1']
#df['SIT']
if name_file == 'DATA':
	df = df[['Field1','SIT']]
	pattern = '[' + punctuation + ']*\s*[Nn]o\s+response.[' + punctuation + ']*'
	df = df[~df['SIT'].str.contains(pattern)]
	#df['SIT'] = df['SIT'].str.replace('á','')
	#df['SIT'] = df['SIT'].str.replace('\n','')
	# keep only five emotions (anger, disgust, fear, joy and sadness)
	df = df[~df['Field1'].str.contains('guilt')]
	df = df[~df['Field1'].str.contains('shame')]
	df['labels'] = pd.Categorical(df['Field1']).codes
	classes = len(pd.Categorical(df['labels']).categories)

	train, test = train_test_split(df, test_size=0.2, random_state=42)

	x_train = np.asarray([sent.lower() for sent in train['SIT']])
	y_train = np.asarray(train['Field1'])
	x_test = np.asarray([sent.lower() for sent in test['SIT']])
	y_test = np.asarray(test['Field1'])
else:
	df = df[~df[1].str.contains("NO RESPONSE")]
	# keep only five emotions (anger, disgust, fear, joy and sadness)
	df = df[~df[0].str.contains('guilt')]
	df = df[~df[0].str.contains('shame')]
	df[2] = pd.Categorical(df[0]).codes
	classes = len(pd.Categorical(df[2]).categories)

	train, test = train_test_split(df, test_size=0.2, random_state=42)

	x_train = np.asarray([sent.lower() for sent in train[0]])
	y_train = np.asarray(train[2])
	x_test = np.asarray([sent.lower() for sent in test[0]])
	y_test = np.asarray(test[2])


for idx in range(np.shape(x_train)[0]):
	x_train[idx] = preprocessing(x_train[idx])

for idx in range(np.shape(x_test)[0]):
	x_test[idx] = preprocessing(x_test[idx])

#for i in range(len(x_train)):
#	pass
#exit()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train)
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)


# get the word to index mapping for input language
word2idx = tokenizer.word_index
idx2word = {y: x for x, y in word2idx.items()}

print('Found %s unique input tokens.' % len(word2idx))


##############################################################################################################3
word2vec = {}
#path = '/home/carolina/corpora/embeddings/glove/glove.42B.300d.txt'
#emb_type = 'combined_class_reg'
#path = '/home/carolina/embeddings/dense_model/emb/last_version/' + emb_type + '.txt'
emb_type = 'out_test'
path = '/home/carolina/embeddings/dense_model/emb/' + emb_type + '.txt'
#emb_type = 'glove'
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
print('Getting training embeddings...')
x_train = get_embeddings(x_train, word2vec, idx2word, 'training')
print('Getting test embeddings...')
x_test = get_embeddings(x_test, word2vec, idx2word, 'test')
print("Time filling embeddings: ", (time.time() - start_time))

print('Starting classification...')
start_time = time.time()
svm_classifier = SVC(kernel='poly', C=1, degree=3)
#svm_classifier = NuSVC(nu=0.03)
svm_classifier.fit(x_train,y_train)
print('Training ended')
print("Time training: ", (time.time() - start_time))


print('-----------------------------------------------------------------')
start_time = time.time()
pred_test = svm_classifier.predict(x_test)
test_accuracy = accuracy_score(y_test, pred_test)
test_f1 = f1_score(y_test, pred_test, average='micro')
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
