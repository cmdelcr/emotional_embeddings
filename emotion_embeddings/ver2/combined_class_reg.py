import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, r2_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from gensim import models

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from DenseModel import DenseModel
from sklearn import preprocessing	

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt

from util_combined import *
import settings

emb_type = 'glove'
type_matrix_emb = 'lexicons' #[lexicons, full]


def create_model(input_shape, output_classification_size):
	input_ = Input(shape=(input_shape,))
	hidden_shared_layer = Dense(100, activation='relu') 
	x1 = hidden_shared_layer(input_)
	hidden_layer = Dense(50, activation='relu') 
	x2 = hidden_layer(x1)
	hidden_layer1 = Dense(10, activation='relu') 
	x2 = hidden_layer1(x2)

	output_regression = Dense(3, activation='linear')(x1)
	output_classification = Dense(output_classification_size, activation='sigmoid')(x2)

	model = Model(inputs=[input_],outputs=[output_regression,output_classification])


	return model

def compile_model(model):
	model.compile(
			loss=['mean_squared_error','categorical_crossentropy'],
			optimizer='adam',#Adam(lr=0.001),
			metrics=[tf.keras.metrics.RootMeanSquaredError(), 'accuracy']
	)
	plot_model(model, to_file='model.png', show_shapes=True)

	return model

def def_class_weight(arr_class_counter):
	arr_class = {}
	for idx in range(len(arr_class_counter)):
		#n_samples / (n_classes * n_samplesj)
		arr_class[idx] = np.sum(arr_class_counter) / (len(arr_class_counter) * arr_class_counter[idx])

	print('class_weight: ', arr_class)
	#exit()
	return arr_class

def train_model(model, x_train, y_vad, y_sub, arr_class_counter, type_lex):
	print('Training model...')

	'''checkpoint_filepath = 'tmp/checkpoint_' + type_lex
	model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
			filepath=checkpoint_filepath,
			save_weights_only=True,
			monitor='val_accuracy',
			mode='max',
			save_best_only=True)


	early_stop = EarlyStopping(monitor='val_accuracy', patience=10)'''

	#r = model.fit(x_train, y_train, validation_data=(x_dev, y_dev), 
	#	batch_size=512, epochs=50, verbose=0, callbacks=[model_checkpoint_callback, early_stop])


	r = model.fit(x_train, 
		[y_vad, y_sub], 
		batch_size=128, 
		epochs=150, 
		#class_weight=[{}, def_class_weight(arr_class_counter)],
		#callbacks=[model_checkpoint_callback, early_stop],
		verbose=1)

	return r

def evaluate_model(model, y_vad, y_sub, plot_results=False):
	results_regression, results_classification = model.evaluate(embedding_matrix, [y_vad, y_sub])
	print("train loss, train rmse:", results_regression)
	print("train loss, train accuracy:", results_classification)

	if plot_results:
		plt.plot(r.history['loss'], label='loss')
		plt.legend()
		plt.show()

		# accuracies
		plt.plot(r.history['accuracy'], label='acc')
		plt.legend()
		plt.show()

	return results_regression, results_classification



print('Loading vad lexicon...')
dict_vad = read_vad_file()
#print('Loading subjectivity_clues lexicon...')
#type_lex = 'sub_clues'
#dict_sub, arr_class_counter = read_subjectivity_clues()
print('Loading emo_lex lexicon...')
type_lex = 'emo_lex'
dict_emo_lex, arr_class_counter = read_subjectivity_clues()

word2vec = read_embeddings(emb_type)
word2idx, vocabulary = getting_lemmas(emb_type, dict_vad, dict_emo_lex, word2vec)

print('-----------------------------------------')
print('***Filling pre-trained embeddings...')
embedding_matrix, vocabulary, y_vad, y_sub = filling_embeddings(
			word2idx, word2vec, vocabulary, dict_vad, dict_emo_lex, emb_type, type_matrix_emb, type_lex)
print('size_embeddings: ', np.shape(embedding_matrix))


print('--------------------------')
print('Creating the model...')
model = create_model(len(embedding_matrix[0]), len(y_sub[0]))
model = compile_model(model)

r = train_model(model, embedding_matrix, y_vad, y_sub, arr_class_counter, type_lex)
#results_reg, results_class = evaluate_model(model, y_train_)
pred_reg, pred_class = model.predict(embedding_matrix)

r2 = r2_score(y_vad, pred_reg)
print('------------------------------------')
print('r2: ', r2)

pred_class = pred_class.round()	
acc = accuracy_score(y_sub, y_pred=pred_class)
print('accuracy: ', acc)


senti_embedding = merge_semantic_end_emotion_embeddings(model, embedding_matrix, type_matrix_emb)
print('Senti embeddings size PCA', np.shape(senti_embedding))
name_file = 'combined_class_reg'

save_senti_embeddings(senti_embedding, vocabulary, name_file, type_matrix_emb)




