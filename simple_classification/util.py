import re
import os
import pandas as pd
from string import punctuation
from nltk import TweetTokenizer

from gensim import models
import settings

import numpy as np

emotions = ['negative', 'positive', 'neutral']
punctuation_list = list(punctuation)

def remove_unecesary_data(sent):
	# remove urls (https?:\/\/\S+) --> for urls with http
	#sent = re.sub(r'https?:\/\/\S+', '', sent)
	#sent = re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", '', sent)
	sent = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','url',sent)
	sent = re.sub(r'#([^\s]+)', r'\1', sent)
	
	#Removes unicode strings like "\u002c" and "x96"
	sent = re.sub(r'(\\u[0-9A-Fa-f]+)',r'', sent)       
	sent = re.sub(r'[^\x00-\x7f]',r'',sent)
	
	#remove numbers
	#sent = ''.join([token for token in sent if not token.isdigit()])

	#removing handles
	sent = re.sub('@[^\s]+','handles',sent)
	
	# remove the symbol from hastag to analize the word
	sent = re.sub(r'#([^\s]+)', r'\1', sent)
	#sent = re.sub(r'#', '', sent)
	#print(sent)

	return sent


def preprocessing(sent):
	sent = remove_unecesary_data(sent)

	tknzr = TweetTokenizer(preserve_case=True, reduce_len=True, strip_handles=True)
	tokens = tknzr.tokenize(sent)

	return [w for w in tokens if w not in punctuation_list]

def read_datasets(dir_datasets):
	datasets = {'train': [], 'dev': [], 'test': []}
	
	for file in os.listdir(dir_datasets):
		if re.sub('.txt', '', file) in datasets.keys():
			key = re.sub('.txt', '', file)
			df = pd.read_csv(dir_datasets + file, sep='\t', header=None)
			df[3] = pd.Categorical(df[1]).codes
			classes = pd.Categorical(df[1]).categories

			for index, row in df.iterrows():
				datasets[key].append((row[3], str(row[2])))
				
	y_train, x_train = zip(*datasets['train'])
	y_dev, x_dev = zip(*datasets['dev'])
	y_test, x_test = zip(*datasets['test'])

	x_train = [preprocessing(sent.lower()) for sent in x_train]
	x_dev = [preprocessing(sent.lower()) for sent in x_dev]
	x_test = [preprocessing(sent.lower()) for sent in x_test]

	#y_train = list(y_train)
	#y_dev = list(y_dev)
	#y_test = list(y_test)

	y_train = np.asarray(y_train, dtype='int32')
	y_dev = np.asarray(y_dev, dtype='int32')
	y_test = np.asarray(y_test, dtype='int32')

	

	return y_train, x_train, y_dev, x_dev, y_test, x_test, classes

def read_embeddings(type_emb='glove'):
	print('\n-----------------------------------------')
	print('Loading embeddings ' + type_emb + '...')
	print('-----------------------------------------')

	word2vec = {}
	if type_emb == 'word2vec':
		word2vec = models.KeyedVectors.load_word2vec_format(settings.dir_embeddings_word2vec, binary=True)
		#word2vec.index_to_key
	else:
		path = settings.dir_embeddings_glove if type_emb == 'glove' else settings.dir_embeddings_numberbatch
		for line in open(path):
			values = line.split()
			word2vec[str(values[0]).lower()] = np.asarray(values[1:], dtype='float32')

	return word2vec		


def read_embeddings_generated(str_dir):
	print('\n-----------------------------------------')
	print('Loading embeddings ' + re.sub(r'\.txt', '', str_dir) + '...')

	word2vec = {}
	path = settings.local_dir_embeddings + 'dense_model/' + str_dir
	for line in open(path):
		values = line.split()
		word2vec[str(values[0]).lower()] = np.asarray(values[1:], dtype='float32')

	return word2vec		