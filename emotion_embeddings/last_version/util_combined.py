import os
import re
from gensim import models
import pandas as pd
from nltk.stem import WordNetLemmatizer
import numpy as np
import settings
import random

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import TruncatedSVD


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


# modifies the last column that represent other_emotion (for those words where is no emotion assigned)
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




def read_emo_lex_file(only_emotions=True, multi_label=True):
	print('Loading emo_lex lexicon...')
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
				if not multi_label:
					continue
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
	

def read_subjectivity_clues(multi_label=True):
	print('Loading subjectivity_clues lexicon...')
	dict_data = {}
	arr_counts = np.zeros(4)
	with open('/home/carolina/corpora/lexicons/subjectivity_clues/subjclueslen1-HLTEMNLP05.tff', 'r') as file:
		for line in file:
			row = line.split()
			key = re.sub(r'word1=', '', str(row[2])).lower()

			if key in dict_data:
				if not multi_label:
					continue
				arr_sub = dict_data[key]
			else:
				arr_sub = def_value(row)

			dict_data[key] = arr_sub
			arr_counts = np.add(arr_counts, arr_sub)
		file.close()

	#print(len(dict_data))
	#print(arr_counts)
	#exit()

	return dict_data, arr_counts


def getting_lemmas(emb_type, dict_vad, dict_sub, dict_emo_lex, word2vec):
	print('Getting lemmas...')
	counter_lem_lex = 0
	counter_word = 0
	counter_word_dict = 0
	word2idx = {}
	vocabulary = []

	lemmatizer = WordNetLemmatizer()
	list_keys = list(word2vec.keys()) if emb_type != 'word2vec' else list(word2vec.key_to_index.keys())
	set_aux = set(dict_vad.keys()).union(set(dict_sub.keys()))
	words_lexicons = dict.fromkeys(list(set_aux.union(set(dict_emo_lex.keys()))), 0)
	
	for key in list_keys:
		if key in words_lexicons:
			vocabulary.append(key)
			counter_word_dict += 1
			word2idx[key] = counter_word
			counter_word += 1
		else:
			lemma = lemmatizer.lemmatize(key)
			if lemma in words_lexicons and lemma not in word2idx:
				counter_lem_lex += 1
				vocabulary.append(key)
				word2idx[key] = counter_word
				counter_word += 1

	print('words in lexicons and ' + emb_type + ': ', counter_word_dict)
	print('lemmas_lexicons: ', counter_lem_lex)
	print('final vocabulary size: ', counter_word)

	return word2idx, vocabulary


def getting_lemmas_two_lex(emb_type, dict_vad, dict_emo_lex, word2vec):
	print('Getting lemmas...')
	counter_lem_lex = 0
	counter_word = 0
	counter_word_dict = 0
	word2idx = {}
	vocabulary = []

	lemmatizer = WordNetLemmatizer()
	list_keys = list(word2vec.keys()) if emb_type != 'word2vec' else list(word2vec.key_to_index.keys())
	words_lexicons = dict.fromkeys(list(set(dict_vad.keys()).union(set(dict_emo_lex.keys()))), 0)
	
	for key in list_keys:
		if key in words_lexicons:
			vocabulary.append(key)
			counter_word_dict += 1
			word2idx[key] = counter_word
			counter_word += 1
		else:
			lemma = lemmatizer.lemmatize(key)
			if lemma in words_lexicons and lemma not in word2idx:
				counter_lem_lex += 1
				vocabulary.append(key)
				word2idx[key] = counter_word
				counter_word += 1

	print('words in lexicons and ' + emb_type + ': ', counter_word_dict)
	print('lemmas_lexicons: ', counter_lem_lex)
	print('final vocabulary size: ', counter_word)

	return word2idx, vocabulary

def filling_embeddings_two_lex(word2idx, word2vec, vocabulary, dict_vad, dict_emo_lex, emb_type, type_matrix_emb, type_lex):
	count_known_words = 0
	count_unknown_words = 0
	count_vad = 0
	count_emo_lex = 0
	y_vad = []
	y_emo_lex = []

	# type_matrix_emb, if lexicons only load the embeddings where is a value in the lexicons, if full
	# load all the embeddings creatings neutral values for words not present in the lexicons
	if type_matrix_emb == 'lexicons':
		embeddings_list = vocabulary
	else:
		set1 = set(vocabulary)
		set2 = set(list(word2vec.keys()) if emb_type != 'word2vec' else list(word2vec.key_to_index.keys()))
		embeddings_list = list(set.union(set1, set2))

	embedding_matrix = np.zeros((len(embeddings_list), 300))
	i = 0
	size_output_emo = random.choice(list(dict_emo_lex.items()))[1]
	#print('size_outpt_class sub: ', len(size_output))
	print('size_outpt_class emo: ', len(size_output_emo))
	for word in embeddings_list:
		embedding_vector = word2vec[word]
		if embedding_vector is None:
			# words not found in embedding index will be initialized with a gaussian distribution.
			embedding_matrix[i] = np.random.uniform(-0.25, 0.25, 300)
			count_unknown_words += 1
		else:
			embedding_matrix[i] = embedding_vector
			count_known_words += 1

		# addings vad values
		if word in dict_vad:
			y_vad.append(dict_vad[word])
			count_vad += 1
		else:
			y_vad.append(np.zeros(3))

		# addings emo_lex values
		if word in dict_emo_lex:
			y_emo_lex.append(dict_emo_lex[word])
			count_emo_lex += 1
		else:
			arr = np.zeros(len(size_output_emo))
			arr[-1] = 1
			y_emo_lex.append(arr)
		i += 1
	y_vad = np.asarray(y_vad, dtype='float32')
	y_emo_lex = np.asarray(y_emo_lex, dtype='int32')

	print('Size words initialized with a gaussian distribution: ', count_unknown_words)
	print('Size words with a value in word2vec: ', count_known_words)
	print('Num of vad values: ', count_vad, ', size: ', np.shape(y_vad))
	print('Num emo_lex: ', count_emo_lex, ', size: ', np.shape(y_emo_lex))

	
	return embedding_matrix, embeddings_list, y_vad, y_emo_lex

def filling_embeddings(word2idx, word2vec, vocabulary, dict_vad, dict_sub, dict_emo_lex, emb_type, type_matrix_emb, type_lex):
	count_known_words = 0
	count_unknown_words = 0
	count_vad = 0
	count_sub = 0
	count_emo_lex = 0
	y_vad = []
	y_sub = []
	y_emo_lex = []

	# type_matrix_emb, if lexicons only load the embeddings where is a value in the lexicons, if full
	# load all the embeddings creatings neutral values for words not present in the lexicons
	if type_matrix_emb == 'lexicons':
		embeddings_list = vocabulary
	else:
		set1 = set(vocabulary)
		set2 = set(list(word2vec.keys()) if emb_type != 'word2vec' else list(word2vec.key_to_index.keys()))
		embeddings_list = list(set.union(set1, set2))

	embedding_matrix = np.zeros((len(embeddings_list), 300))
	i = 0
	size_output = random.choice(list(dict_sub.items()))[1]
	size_output_emo = random.choice(list(dict_emo_lex.items()))[1]
	print('size_outpt_class sub: ', len(size_output))
	print('size_outpt_class emo: ', len(size_output_emo))
	for word in embeddings_list:
		embedding_vector = word2vec[word]
		if embedding_vector is None:
			# words not found in embedding index will be initialized with a gaussian distribution.
			embedding_matrix[i] = np.random.uniform(-0.25, 0.25, 300)
			count_unknown_words += 1
		else:
			embedding_matrix[i] = embedding_vector
			count_known_words += 1

		# addings vad values
		if word in dict_vad:
			y_vad.append(dict_vad[word])
			count_vad += 1
		else:
			y_vad.append(np.zeros(3))

		# addings sub_clue values
		if word in dict_sub:
			y_sub.append(dict_sub[word])
			count_sub += 1
		else:
			arr = np.zeros(len(size_output))
			arr[-1] = 1
			y_sub.append(arr)

		# addings emo_lex values
		if word in dict_emo_lex:
			y_emo_lex.append(dict_emo_lex[word])
			count_emo_lex += 1
		else:
			arr = np.zeros(len(size_output_emo))
			arr[-1] = 1
			y_emo_lex.append(arr)
		i += 1
	y_vad = np.asarray(y_vad, dtype='float32')
	y_sub = np.asarray(y_sub, dtype='int32')
	y_emo_lex = np.asarray(y_emo_lex, dtype='int32')

	print('Size words initialized with a gaussian distribution: ', count_unknown_words)
	print('Size words with a value in word2vec: ', count_known_words)
	print('Num of vad values: ', count_vad, ', size: ', np.shape(y_vad))
	print('Num sub_clues: ', count_sub, ', size: ', np.shape(y_sub))
	print('Num emo_lex: ', count_emo_lex, ', size: ', np.shape(y_emo_lex))

	
	return embedding_matrix, embeddings_list, y_vad, y_sub, y_emo_lex


def relu(arr):
	return np.maximum(0, arr)



def merge_semantic_end_emotion_embeddings(model, embedding_matrix, type_matrix_emb, print_sizes=True):
	print('merging embeddings ...')
	if print_sizes:
		print('Matrix input_to_dense: ', np.shape(model.layers[1].get_weights()[0]))
		print('Bias input_to_dense: ', np.shape(model.layers[1].get_weights()[1]))
		#print('Matrix dense_to_output: ', np.shape(model.layers[2].get_weights()[0]))
		#print('Bias dense_to_output', np.shape(model.layers[2].get_weights()[1]))
		

	input_matrix_dense = model.layers[1].get_weights()[0]
	input_bias_dense = model.layers[1].get_weights()[1]
	#output_matrix_dense = model.layers[2].get_weights()[0]
	#output_bias_dense = model.layers[2].get_weights()[1]

	print('------------------------------------------')
	senti_embedding = np.dot(embedding_matrix, input_matrix_dense) + input_bias_dense
	#senti_embedding = np.apply_along_axis(relu, 0, senti_embedding)
	senti_embedding = np.hstack((embedding_matrix, senti_embedding))
	
	print('apply_pca')
	
	if type_matrix_emb == 'full':
		n = senti_embedding.shape[0] # how many rows we have in the dataset
		chunk_size = 2000 # how many rows we feed to IPCA at a time, the divisor of n
		ipca = IncrementalPCA(n_components=300, batch_size=16)

		for i in range(0, n//chunk_size):
			ipca.partial_fit(senti_embedding[i*chunk_size : (i+1)*chunk_size])

		return ipca.transform(senti_embedding)#, senti_embedding_no_pca
	else:
		pca = PCA(300)
		return pca.fit_transform(senti_embedding)#, senti_embedding_no_pca


def merge_semantic_end_emotion_embeddings_emb_layer(model, embedding_matrix, type_matrix_emb, word2idx):
	print('merging embeddings ...')

	embeddings = model.layers[1].get_weights()[0]
	words_embeddings = {w:embeddings[idx] for w, idx in word2idx.items()}

	return words_embeddings




def save_senti_embeddings_emb_layer(senti_embedding, labels_, labels, name_file, type_matrix_emb):
	dir_name = settings.local_dir_embeddings + 'dense_model/emb/last_version'

	if not os.path.exists(dir_name):
		os.makedirs(dir_name)

	i = 0
	with open(os.path.join(dir_name, name_file + '.txt'), 'w') as f:
		for key, value in senti_embedding.items():
			if labels_[i] in labels:
				f.write(key + " ")
				np.savetxt(f, fmt='%.6f', X=np.atleast_2d(value), delimiter=' ', newline='\n')
			i += 1
		f.close()


def save_senti_embeddings(senti_embedding, labels_, labels, name_file, type_matrix_emb):
	dir_name = settings.local_dir_embeddings + 'dense_model/emb/last_version'

	if not os.path.exists(dir_name):
		os.makedirs(dir_name)

	i = 0
	with open(os.path.join(dir_name, name_file + '.txt'), 'w') as f:
		mat = np.matrix(senti_embedding)
		for w_vec in mat:
			if labels_[i] in labels:
				f.write(labels_[i].replace(" ", "_" ) + " ")
				np.savetxt(f, fmt='%.6f', X=w_vec)
			i += 1
		f.close()



