import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from gensim import models


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from DenseModel import DenseModel
from sklearn import preprocessing

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt

from util import *
import settings


embedding_dim = 300
dim_arr = [100]#[10, 30, 50, 100, 200, 300]
arr_epochs = [500]#[100, 200, 300, 400, 500]
arr_activation_functions = ['tanh']#['tanh', 'relu', 'sigmoid', 'exponential']
arr_type_matrix_emb = ['full']


def create_model(input_shape, num_units, activation_function):
	input_ = Input(shape=(input_shape,))
	dense = Dense(num_units, activation='tanh', 
		kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))
	x1 = dense(input_)
	output = Dense(3, activation='linear')(x1)

	model = Model(inputs=input_, outputs=output)

	return model

def compile_model(model, loss_='mean_squared_error', optimizer_='adam'):
	# regular categorical_crossentropy requires one_hot_encoding for the targets, 
	#sparse_categorical_crossentropy is used to don't use the conversion
	model.compile(
			loss=loss_,
			optimizer=optimizer_,#Adam(lr=0.001),
			metrics=['accuracy']
	)

	return model

def train_model(model, x_train, y_train, batch_size_=512, epochs_=200, verbose=0):
	print('Training model...')
	r = model.fit(x_train, 
		y_train, 
		batch_size=batch_size_, 
		epochs=epochs_, 
		verbose=0)

	return r

def evaluate_model(model, y_train, plot_results=False):
	results = model.evaluate(embedding_matrix, y_train)
	print("train loss, train acc:", results)

	if plot_results:
		plt.plot(r.history['loss'], label='loss')
		plt.legend()
		plt.show()

		# accuracies
		plt.plot(r.history['accuracy'], label='acc')
		plt.legend()
		plt.show()


def merge_semantic_end_emotion_embeddings(model, embedding_matrix, type_matrix_emb, act='tanh', apply_pca=True, print_sizes=False):
	print('merging embeddings ...')
	if print_sizes:
		print('Matrix input_to_dense: ', np.shape(model.layers[1].get_weights()[0]))
		print('Bias input_to_dense: ', np.shape(model.layers[1].get_weights()[1]))
		print('Matrix dense_to_output: ', np.shape(model.layers[2].get_weights()[0]))
		print('Bias dense_to_output', np.shape(model.layers[2].get_weights()[1]))

	input_matrix_dense = model.layers[1].get_weights()[0]
	input_bias_dense = model.layers[1].get_weights()[1]
	output_matrix_dense = model.layers[2].get_weights()[0]
	output_bias_dense = model.layers[2].get_weights()[1]

	senti_embedding = embedding_matrix
	#print('^^^^')
	#print(np.shape(senti_embedding))
	senti_embedding = np.dot(embedding_matrix, input_matrix_dense) + input_bias_dense
	#print(np.shape(senti_embedding))
	senti_embedding = np.apply_along_axis(np.tanh, 0, senti_embedding)
	#print(np.shape(senti_embedding))
	senti_embedding_no_pca = np.hstack((embedding_matrix, senti_embedding))
	#print(np.shape(senti_embedding))
	#exit()

	#if apply_pca:
	print('apply_pca')
	#TruncatedSVD, LinearDiscriminantAnalysis, Isomap, LocallyLinearEmbedding

	if type_matrix_emb == 'full':
		n = senti_embedding_no_pca.shape[0] # how many rows we have in the dataset
		chunk_size = 5000 # how many rows we feed to IPCA at a time, the divisor of n
		ipca = IncrementalPCA(n_components=300, batch_size=16)

		for i in range(0, n//chunk_size):
			ipca.partial_fit(senti_embedding_no_pca[i*chunk_size : (i+1)*chunk_size])

		return ipca.transform(senti_embedding_no_pca)#, senti_embedding_no_pca
	else:
		return pca.fit_transform(senti_embedding_no_pca)#, senti_embedding_no_pca



#------------------------------------------------------------------------------------------------------------------
dict_data = read_vad_file()
print('NRC-VAD size: ', len(dict_data))


for emb_type in settings.embedding_type:
	word2vec = read_embeddings(emb_type)
	word2idx, vocabulary, vad_value = getting_lemmas(emb_type, dict_data, word2vec)
	
	#for type_matrix_emb in arr_type_matrix_emb:
	type_matrix_emb = 'full'
	set1 = set(vocabulary)
	set2 = set(list(word2vec.keys()) if emb_type != 'word2vec' else list(word2vec.key_to_index.keys()))
	set_all = set.union(set1, set2)
	embedding_matrix_full = np.zeros((len(set_all), embedding_dim+3))
	#embedding_matrix = np.zeros((len(vocabulary), embedding_dim+3))
	voc = []
	i = 0
	for word in set_all:
		voc.append(word)
		embedding_vector = word2vec[word]
		if embedding_vector is None:
			# words not found in embedding index will be initialized with a gaussian distribution.
			embedding_matrix_full[i][0:300] = np.random.uniform(-0.25, 0.25, embedding_dim)
			embedding_matrix_full[i][300:303] = np.asarray([0.5, 0.5, 0.5]) if word not in dict_data else dict_data[word]
			#count_unknown_words += 1
		else:
			embedding_matrix_full[i][0:300] = embedding_vector
			embedding_matrix_full[i][300:303] = np.asarray([0.5, 0.5, 0.5]) if word not in dict_data else dict_data[word]
			#count_known_words += 1
		i += 1


	#embedding_matrix = np.zeros((len(vocabulary), embedding_dim+3))


	dir_name = settings.local_dir_embeddings + 'concatenate_vad'
	if not os.path.exists(dir_name):
		os.makedirs(dir_name)
	with open(os.path.join(dir_name, 'concatenate_vad_%d.txt' % 303), 'w') as f:
		i = 0
		mat = np.matrix(embedding_matrix_full)
		for w_vec in mat:
			f.write(voc[i].replace(" ", "_" ) + " ")
			np.savetxt(f, fmt='%.6f', X=w_vec)
			i += 1
		f.close()