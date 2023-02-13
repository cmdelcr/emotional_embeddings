import os
import re
from gensim import models
import pandas as pd
from nltk.stem import WordNetLemmatizer
import numpy as np
import settings


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

def read_vad_file():
	dict_data = {}
	df = pd.read_csv(settings.input_dir_lexicon_vad + 'NRC-VAD-Lexicon/NRC-VAD-Lexicon.txt', 
				keep_default_na=False, header=None, sep='\t')
	for index, row in df.iterrows(): #V, A, D
		dict_data[str(row[0]).lower()] = [float(row[1]), float(row[2]), float(row[3])]

	return dict_data


def read_emo_lex_file(only_emotions):
	df_emo_lex = pd.read_csv('/home/carolina/corpora/lexicons/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt', 
			keep_default_na=False, header=None, sep='\t')

	arr_emotions = ['anger', 'fear', 'anticipation', 'trust', 'surprise', 'sadness', 'joy', 'disgust', 'negative', 'positive', 'no_emo_pol']
	if only_emotions:
		for i in range(2):
			arr_emotions = list(np.delete(arr_emotions, 8, axis=0))

	arr_counter = np.zeros(8) if only_emotions else np.zeros(10)
	dict_emo_lex = {}
	for index, row in df_emo_lex.iterrows():
		if not (str(row[1]) == 'negative' or str(row[1]) == 'positive') or not only_emotions: 
			if str(row[0]) in dict_emo_lex:
				arr_emo_lex = dict_emo_lex[str(row[0])]
			else:
				arr_emo_lex = np.zeros(9) if only_emotions else np.zeros(11)
			idx = arr_emotions.index(str(row[1]))
			if int(row[2]) == 1:
				#print(arr_counter)
				arr_counter[idx] = arr_counter[idx] + 1
			arr_emo_lex[idx] = int(row[2])
			dict_emo_lex[str(row[0])] = arr_emo_lex

	return verify_emo_pol(dict_emo_lex, arr_counter)


def verify_emo_pol(dict_emo_lex, arr_counter):
	counter = 0
	for key, value in dict_emo_lex.items():
		if not value.any():
			value[-1] = 1
			dict_emo_lex[key] = value
			counter += 1

	aux = np.zeros(1)
	aux[0] = counter
	arr_counter = np.append(arr_counter, aux, axis=0)

	return dict_emo_lex, arr_counter

def def_value(row):
	arr_value = np.zeros(4)

	# strongly_positive, weakly_positive, strongly_negative, and weakly_negative
	if 'positive' in str(row[5]):
		if 'strong' in str(row[0]):
			arr_value[0] = 1
		else:
			arr_value[1] = 1
	else:
		if 'strong' in str(row[0]):
			arr_value[2] = 1
		else:
			arr_value[3] = 1

	return arr_value

def def_values_keys(row, key, dict_data):
	if key in dict_data:
		if 'positive' in str(row[5]) and 'strong' in str(row[0]) and dict_data[key][2] == 1:
			print('Error, different polarity (' + key + '): ', row[0], ', ', row[5])
			print(dict_data[key])
		if 'positive' in str(row[5]) and 'weak' in str(row[0]) and dict_data[key][2] == 1:
			print('Error, different polarity (' + key + '): ', row[0], ', ', row[5])
			print(dict_data[key])
		if 'negative' in str(row[5]) and 'strong' in str(row[0]) and dict_data[key][1] == 1:
			print('Error, different polarity (' + key + '): ', row[0], ', ', row[5])
			print(dict_data[key])
		if 'negative' in str(row[5]) and 'weak' in str(row[0]) and dict_data[key][1] == 1:
			print('Error, different polarity (' + key + '): ', row[0], ', ', row[5])
			print(dict_data[key])
	

def read_subjectivity_clues():
	dict_data = {}
	with open('/home/carolina/corpora/lexicons/subjectivity_clues/subjclueslen1-HLTEMNLP05.tff', 'r') as file:
		for line in file:
			row = line.split()
			key = re.sub(r'word1=', '', str(row[2])).lower()
			#def_values_keys(row, key, dict_data)
			dict_data[key] = def_value(row)
		file.close()

	print(np.shape(dict_data))
	exit()

	return dict_data

def getting_lemmas(emb_type, vad, word2vec):
	counter_lem = 0
	counter_word_dict = 0
	counter_word = 0
	word2idx = {}
	vad_value = []
	vocabulary = []

	lemmatizer = WordNetLemmatizer()
	list_keys = list(word2vec.keys()) if emb_type != 'word2vec' else list(word2vec.key_to_index.keys())

	for key in list_keys:
		if key in vad:
			vocabulary.append(key)
			counter_word_dict += 1
			word2idx[key] = counter_word
			counter_word += 1
			vad_value.append(vad[key])
		else:
			lemma = lemmatizer.lemmatize(key)
			if lemma in vad and lemma not in word2idx:
				counter_lem += 1
				vocabulary.append(key)
				word2idx[key] = counter_word
				counter_word += 1
				vad_value.append(vad[lemma])

	print('words in vad and ' + emb_type + ': ', counter_word_dict)
	print('lemmas: ', counter_lem)
	print('final vocabulary size: ', counter_word)

	return word2idx, vocabulary, vad_value

def filling_embeddings_vad_values(word2idx, word2vec, vocabulary, embedding_dim, emb_type, y_train):
	print('***Filling pre-trained embeddings...')
	count_known_words = 0
	count_unknown_words = 0
	counter_stop_words = 0

	embedding_matrix = np.zeros((len(vocabulary), embedding_dim))
	for word, i in word2idx.items():
		embedding_vector = word2vec[word]
		if embedding_vector is None:
			# words not found in embedding index will be initialized with a gaussian distribution.
			embedding_matrix[i] = np.random.uniform(-0.25, 0.25, embedding_dim)
			count_unknown_words += 1
		else:
			embedding_matrix[i] = embedding_vector
			count_known_words += 1
	y_train = np.asarray(y_train, dtype='float32')

	
	return embedding_matrix, vocabulary, y_train

def filling_embeddings_full_matrix(word2idx, word2vec, vocabulary, embedding_dim, emb_type, y_train):
	print('***Filling pre-trained embeddings...')
	count_known_words = 0
	count_unknown_words = 0
	counter_stop_words = 0

	set1 = set(vocabulary)
	set2 = set(list(word2vec.keys()) if emb_type != 'word2vec' else list(word2vec.key_to_index.keys()))
	set_all = set.union(set1, set2)
	embedding_matrix = np.zeros((len(set_all), embedding_dim))
	i = 0
	y_train_ = []
	vocabulary_ = []
	for word in list(word2vec.keys()) if emb_type != 'word2vec' else list(word2vec.key_to_index.keys()):
		embedding_matrix[i] = word2vec[word]
		if word in vocabulary:
			y_train_.append(y_train[vocabulary.index(word)])
		else:
			y_train_.append([0, 0, 0])
		vocabulary_.append(word)
		i += 1
	
	for word in list(set1 - set2):
		embedding_matrix[i] = np.random.uniform(-0.25, 0.25, embedding_dim)
		vocabulary_.append(word)
		y_train_.append(y_train[vocabulary.index(word)])
		i += 1
	vocabulary = vocabulary_
	y_train = np.asarray(y_train_, dtype='float32')
	print('total_words: ', len(vocabulary))
	print('total_vad_values: ', len(y_train))

	
	return embedding_matrix, vocabulary, y_train

def filling_embeddings(word2idx, word2vec, vocabulary, embedding_dim, emb_type, type_matrix_emb, y_train):
	print('***Filling pre-trained embeddings...')
	count_known_words = 0
	count_unknown_words = 0
	counter_stop_words = 0

	if type_matrix_emb == 'vad': 
		embedding_matrix = np.zeros((len(vocabulary), embedding_dim))
		for word, i in word2idx.items():
			embedding_vector = word2vec[word]
			if embedding_vector is None:
				# words not found in embedding index will be initialized with a gaussian distribution.
				embedding_matrix[i] = np.random.uniform(-0.25, 0.25, embedding_dim)
				count_unknown_words += 1
			else:
				embedding_matrix[i] = embedding_vector
				count_known_words += 1
		y_train = np.asarray(y_train, dtype='float32')
		print(np.shape(embedding_matrix))
		exit()
	else:
		set1 = set(vocabulary)
		set2 = set(list(word2vec.keys()) if emb_type != 'word2vec' else list(word2vec.key_to_index.keys()))
		set_all = set.union(set1, set2)
		embedding_matrix = np.zeros((len(set_all), embedding_dim))
		i = 0
		y_train_ = []
		vocabulary_ = []
		for word in list(word2vec.keys()) if emb_type != 'word2vec' else list(word2vec.key_to_index.keys()):
			embedding_matrix[i] = word2vec[word]
			if word in vocabulary:
				y_train_.append(y_train[vocabulary.index(word)])
			else:
				y_train_.append([0, 0, 0])
			vocabulary_.append(word)
			i += 1
		
		for word in list(set1 - set2):
			embedding_matrix[i] = np.random.uniform(-0.25, 0.25, embedding_dim)
			vocabulary_.append(word)
			y_train_.append(y_train[vocabulary.index(word)])
			i += 1
		vocabulary = vocabulary_
		y_train = np.asarray(y_train_, dtype='float32')
		print('total_words: ', len(vocabulary))
		print('total_vad_values: ', len(y_train))

	
	return embedding_matrix, vocabulary, y_train

def save_senti_embeddings(senti_embedding, labels, labels_, name_file, type_matrix_emb):
	dir_name = settings.local_dir_embeddings + 'dense_model/emb'

	if not os.path.exists(dir_name):
		os.makedirs(dir_name)
	with open(os.path.join(dir_name, name_file + '.txt'), 'w') as f:
		i = 0
		mat = np.matrix(senti_embedding)
		for w_vec in mat:
			if labels_[i] in labels:
				f.write(labels_[i].replace(" ", "_" ) + " ")
				np.savetxt(f, fmt='%.6f', X=w_vec)
			i += 1
		f.close()

	if type_matrix_emb == 'full':
		with open(os.path.join(dir_name, name_file + '_full_matrix.txt'), 'w') as f:
			i = 0
			mat = np.matrix(senti_embedding)
			for w_vec in mat:
				f.write(labels_[i].replace(" ", "_" ) + " ")
				np.savetxt(f, fmt='%.6f', X=w_vec)
				i += 1
			f.close()


