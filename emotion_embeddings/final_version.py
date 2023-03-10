import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from gensim import models

import tensorflow as tf
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
arr_epochs = [300]#[100, 200, 300, 400, 500]
arr_activation_functions = ['relu']#['tanh', 'relu', 'sigmoid', 'exponential']
arr_type_matrix_emb = ['full']


def create_model(input_shape, num_units, activation_function):
	input_ = Input(shape=(input_shape,), name='input_layer')
	hidden_shared_layer = Dense(200, activation='relu', name='hidden_shared_layer')#, kernel_initializer='he_normal') 
	x1 = hidden_shared_layer(input_)
	#dense = Dense(num_units, activation='relu') 
		#activation='tanh', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))
	#x1 = dense(input_)

	#layer regression vad
	hidden_layer_vad = Dense(300, activation='relu', name='hidden_layer_vad_1')
	x_vad = hidden_layer_vad(x1)
	#hidden_layer_vad = Dense(200, activation='relu', name='hidden_layer_vad_2')
	#x_vad = hidden_layer_vad(x_vad)



	output = Dense(3, activation='linear')(x_vad)

	model = Model(inputs=input_, outputs=output)

	return model

def compile_model(model, loss_='mean_squared_error', optimizer_='adam'):
	# regular categorical_crossentropy requires one_hot_encoding for the targets, 
	#sparse_categorical_crossentropy is used to don't use the conversion
	model.compile(
			loss='mean_squared_error',
			optimizer=Adam(learning_rate=0.001),
			metrics=[tf.keras.metrics.RootMeanSquaredError()]
	)

	return model

def train_model(model, x_train, y_train, batch_size_=512, epochs_=200, verbose=0):
	print('Training model...')
	r = model.fit(x_train, 
		y_train, 
		batch_size=128, 
		epochs=100, 
		verbose=1)

	return r

def evaluate_model(model, y_train, plot_results=False):
	results = model.evaluate(embedding_matrix, y_train)
	print("train loss, train rmse:", results)

	if plot_results:
		plt.plot(r.history['loss'], label='loss')
		plt.legend()
		plt.show()

		# accuracies
		plt.plot(r.history['accuracy'], label='acc')
		plt.legend()
		plt.show()
	return results

def relu(arr):
	return np.maximum(0, arr)


def merge_semantic_end_emotion_embeddings(model, embedding_matrix, act='tanh', apply_pca=True, print_sizes=True):
	print('merging embeddings ...')
	if print_sizes:
		print('Matrix input_to_dense: ', np.shape(model.layers[1].get_weights()[0]))
		print('Bias input_to_dense: ', np.shape(model.layers[1].get_weights()[1]))
		print('Matrix dense_to_output: ', np.shape(model.layers[2].get_weights()[0]))
		print('Bias dense_to_output', np.shape(model.layers[2].get_weights()[1]))
		print('Matrix dense_to_output: ', np.shape(model.layers[3].get_weights()[0]))
		print('Bias dense_to_output', np.shape(model.layers[3].get_weights()[1]))

	input_matrix_dense = model.layers[1].get_weights()[0]
	input_bias_dense = model.layers[1].get_weights()[1]
	output_matrix_dense = model.layers[2].get_weights()[0]
	output_bias_dense = model.layers[2].get_weights()[1]

	print('^^^^')
	#print(np.shape(senti_embedding))
	#print('size of senti_embeddings: ', np.shape(input_matrix_dense))
	senti_embedding = np.dot(embedding_matrix, model.layers[3].get_weights()[0]) + np.shape(model.layers[3].get_weights()[1])
	#print('size after dot product: ', np.shape(senti_embedding))
	#print('before apply relu')
	#print(senti_embedding[0])
	senti_embedding = np.apply_along_axis(relu, 0, senti_embedding)
	senti_embedding = np.hstack((embedding_matrix, senti_embedding))
	#print('after apply relu')
	#print('size after appy tanh', np.shape(senti_embedding))
	
	#senti_embedding = np.hstack((embedding_matrix, senti_embedding))
	#print('size after stack', np.shape(senti_embedding_no_pca))
	#exit()

	#if apply_pca:
	'''print('apply_pca')
	print(type_matrix_emb)
	#TruncatedSVD, LinearDiscriminantAnalysis, Isomap, LocallyLinearEmbedding

	if type_matrix_emb == 'full':
		n = senti_embedding.shape[0] # how many rows we have in the dataset
		chunk_size = 2000 # how many rows we feed to IPCA at a time, the divisor of n
		ipca = IncrementalPCA(n_components=300, batch_size=16)

		for i in range(0, n//chunk_size):
			ipca.partial_fit(senti_embedding[i*chunk_size : (i+1)*chunk_size])

		return ipca.transform(senti_embedding)#, senti_embedding_no_pca
	else:
		pca = PCA(300)
		return pca.fit_transform(senti_embedding)#, senti_embedding_no_pca'''
	return senti_embedding

def getting_lemmas_(emb_type, vad, word2vec):
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
			#else:
				#print(key)
				#exit()

	print('words in vad and ' + emb_type + ': ', counter_word_dict)
	print('lemmas: ', counter_lem)
	print('final vocabulary size: ', counter_word)
	#exit()
	return word2idx, vocabulary, vad_value


#------------------------------------------------------------------------------------------------------------------
dict_data = read_vad_file()


print('NRC-VAD size: ', len(dict_data))

activation_function = 'tanh'
for emb_type in settings.embedding_type:
	word2vec = read_embeddings(emb_type)
	word2idx, vocabulary, y_train = getting_lemmas_(emb_type, dict_data, word2vec)

	minmax_scale = preprocessing.MinMaxScaler(feature_range=(-1, 1))
	y_train = minmax_scale.fit_transform(y_train)

	#for type_matrix_emb in arr_type_matrix_emb:
	type_matrix_emb = 'vad'
	print('\tType matrix emp: ', type_matrix_emb)

	if type_matrix_emb == 'vad':
		embedding_matrix, vocabulary_, y_train_ = filling_embeddings_vad_values(word2idx, word2vec, vocabulary, embedding_dim, emb_type, y_train)
	else:
		embedding_matrix, vocabulary_, y_train_ = filling_embeddings_full_matrix(word2idx, word2vec, vocabulary, embedding_dim, emb_type, y_train)
	print('Embeddings matrix shape: ', embedding_matrix.shape)

	for embedding_dimention in dim_arr:
		print('--------------------------')
		print('\nFor hidden size: ',  embedding_dimention)
		print('Creating the model...')
		for act in arr_activation_functions:
			#	print('\t   Activation: ', act)
			for epoch in arr_epochs:
				print('\t  Epochs: ', epoch)
				#model = DenseModel(embedding_dimention, act)
				

				model = create_model(len(embedding_matrix[0]), embedding_dimention, act)
				model = compile_model(model)
				r = train_model(model, embedding_matrix, y_train_, epochs_=epoch)
				results = evaluate_model(model, y_train_)
				pred = model.predict(embedding_matrix)
				r2 = r2_score(y_train_, pred)
				mse = mean_squared_error(y_train_, pred)
				print('r2: ', r2)
				print('mse: ', mse)
				#exit()

				#model.
				senti_embedding = merge_semantic_end_emotion_embeddings(model, embedding_matrix, act)
				print(senti_embedding.shape)
				#print(senti_embedding_.shape)
				#exit()
				print('Senti embeddings size PCA', np.shape(senti_embedding))
				#print('Senti embeddings size no PCA', np.shape(senti_embedding_))
				print('----------------------------------------')

				#full_mms_dot_product_hstack_plus_bias_relu_pca
				'''name_file = 'sent_emb_' + emb_type + '_' + type_matrix_emb 	+ '_mms_dot_product_hstack_plus_bias_relu_pca'
				print(name_file)
				with open(os.path.join('/home/carolina/embeddings/dense_model/emb/results_training', name_file + '.txt'), 'w') as f:
					f.write('mean_squared_error: %.6f\nroot_mean_squared_error: %.6f\nr2_score: %.6f' % 
						(results[0], results[1], r2))
					f.close()'''

				name_file = 'out_test.xt'

				save_senti_embeddings(senti_embedding, vocabulary, vocabulary_, name_file, type_matrix_emb)
				#name_file = 'sent_emb_' + emb_type + '_' + str(embedding_dimention) + '_' + act + '_e'+ str(epoch) + '_nopca_' + type_matrix_emb + '.txt'
				#save_senti_embeddings(senti_embedding_, vocabulary, vocabulary_, name_file)
			print('..............................')


	print('------------------------------------------------------------------------------------------------------')
	print('------------------------------------------------------------------------------------------------------\n')