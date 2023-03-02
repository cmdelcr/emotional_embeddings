import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from gensim import models

import tensorflow as tf
from tensorflow.math import confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.backend import epsilon
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.metrics import binary_crossentropy
#from DenseModel import DenseModel
from sklearn import preprocessing	

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt

from util_combined import *
import settings

emb_type = 'glove'
type_matrix_emb = 'lexicons' #[lexicons, full]
out_type_matrix_emb = 'lexicons' #[lexicons, full]
class_weights = {}

#weighted categorical cross entropy
class Custom_Weighted_CE_Loss(tf.keras.losses.Loss):
	def __init__(self):
		super().__init__()

	def __init__(self, class_weights):
		super().__init__()
		self.class_weights = class_weights

	def call(self, y_true, y_pred):		
		log_y_pred = tf.math.log(y_pred + epsilon())
		y_true = tf.cast(y_true, dtype=tf.float32)
		elements = tf.math.multiply_no_nan(x=log_y_pred, y=y_true + epsilon())
		#elements = tf.math.multiply_no_nan(x=elements, y=self.class_weights)
		loss = -tf.reduce_mean(tf.reduce_sum(elements,axis=1))

		return loss

def def_weigths(arr_class_counter):
	positive_weights = {}
	negative_weights = {}

	for idx in range(len(arr_class_counter)):
		positive_weights[idx] = np.sum(arr_class_counter) / (len(arr_class_counter) * arr_class_counter[idx])
		negative_weights[idx] = np.sum(arr_class_counter) / ((len(arr_class_counter) * np.sum(arr_class_counter)) -arr_class_counter[idx])
		
	class_weights['positive_weights'] = positive_weights
	class_weights['negative_weights'] = negative_weights

	return class_weights


def def_class_weight(arr_class_counter):
	arr_class = {}
	for idx in range(len(arr_class_counter)):
		#n_samples / (n_classes * n_samplesj)
		arr_class[idx] = np.sum(arr_class_counter) / (len(arr_class_counter) * arr_class_counter[idx])

	print('class_weight: ', arr_class)
	#exit()
	return arr_class

#weighted binary cross entropy
class Custom_Loss(tf.keras.losses.Loss):
	def __init__(self):
		super().__init__()

	def __init__(self, class_weights):
		super().__init__()
		self.class_weights = class_weights

	def call(self, y_true, y_pred):		
		loss = float(0)
		y_true = tf.cast(y_true, dtype=tf.float32)
		bce = binary_crossentropy(y_true, y_pred)
		weighted_bce = tf.reduce_mean(bce * self.class_weights)

		return weighted_bce

def create_model(input_shape, output_classification_size_emo):
	input_ = Input(shape=(input_shape,), name='input_layer')
	hidden_shared_layer = Dense(200, activation='relu', name='hidden_shared_layer')#, kernel_initializer='he_normal') 
	x1 = hidden_shared_layer(input_)

	#layer regression vad
	hidden_layer_vad = Dense(100, activation='relu', name='hidden_layer_vad_1')
	x_vad = hidden_layer_vad(x1)
	#hidden_layer_vad = Dense(200, activation='relu', name='hidden_layer_vad_2')
	#x_vad = hidden_layer_vad(x_vad)

	# layers classification sub_clues
	'''hidden_layer_sub = Dense(80, activation='tanh', name='hidden_layer_sub',
		kernel_regularizer=regularizers.l2(0.0001), bias_regularizer=regularizers.l2(0.0001)) 
	x_sub = hidden_layer_sub(x1)'''

	#layer classification_emo_lex
	hidden_layer_emo = Dense(30, activation='relu', name='hidden_layer_emo_1')#, 
		#kernel_regularizer=regularizers.l2(0.0001), bias_regularizer=regularizers.l2(0.0001)) 
	x_emo = hidden_layer_emo(x1)
	#hidden_layer_emo_2 = Dense(10, activation='relu', name='hidden_layer_emo_2')#, 
		#kernel_regularizer=regularizers.l2(0.0001), bias_regularizer=regularizers.l2(0.0001)) 
	#x_emo = hidden_layer_emo_2(x_emo)

	output_regression = Dense(3, activation='linear', name='output_reg_vad')(x_vad)
	#output_class_sub = Dense(output_classification_size, activation='sigmoid', name='output_class_sub')(x_sub)
	output_class_emo_lex = Dense(output_classification_size_emo, activation='sigmoid', name='output_class_emo_lex')(x_emo)

	model = Model(inputs=[input_],outputs=[output_regression, output_class_emo_lex])


	return model

def compile_model(model, arr_class_counter_emo, multi_label_emo):
	
	model.compile(
			loss=['mean_squared_error', 
					#'categorical_crossentropy' if not multi_label else BinaryCrossentropy(),
					'categorical_crossentropy' if not multi_label_emo else BinaryCrossentropy()
					#Custom_Loss(list(def_class_weight(arr_class_counter_emo).values()))#'categorical_crossentropy' if not multi_label_emo else 'binary_crossentropy'
					],
			#Custom_Weighted_CE_Loss(def_class_weight(arr_class_counter))], # 'categorical_crossentropy'
			optimizer='adam',#Adam(learning_rate=0.001),#'adam',
			metrics=[tf.keras.metrics.RootMeanSquaredError(), Accuracy(name='acc_emo_lex')]
	)
	plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

	return model


def train_model(model, x_train, y_vad, y_emo_lex, type_lex):
	print('Training model...')


	r = model.fit(x_train, 
		[y_vad, y_emo_lex], 
		batch_size=128, 
		epochs=100,#150 
		#class_weight=[{}, def_class_weight(arr_class_counter)],
		#callbacks=[model_checkpoint_callback, early_stop],
		verbose=0)

	return r

def evaluate_model(model, y_vad, y_emo_lex, plot_results=False):
	results_regression, results_class_emo = model.evaluate(embedding_matrix, [y_vad, y_emo_lex])
	print("train loss, train rmse:", results_regression)
	print("Emo_lex: train loss, train accuracy:", results_class_emo)

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
type_lex = 'emo_lex'
multi_label_emo = True
dict_emo_lex, arr_class_counter_emo = read_emo_lex_file(only_emotions=True, multi_label=multi_label_emo)

word2vec = read_embeddings(emb_type)
word2idx, vocabulary = getting_lemmas_two_lex(emb_type, dict_vad, dict_emo_lex, word2vec)

print('-----------------------------------------')
print('***Filling pre-trained embeddings...')
embedding_matrix, vocabulary_, y_vad, y_emo_lex = filling_embeddings_two_lex(
			word2idx, word2vec, vocabulary, dict_vad, dict_emo_lex, emb_type, type_matrix_emb, type_lex)
print('size_embeddings: ', np.shape(embedding_matrix))


#minmax_scale = preprocessing.MinMaxScaler(feature_range=(-1, 1))
#y_vad = minmax_scale.fit_transform(y_vad)


print('--------------------------')
print('Creating the model...')
model = create_model(len(embedding_matrix[0]), len(y_emo_lex[0]))
model = compile_model(model, arr_class_counter_emo, multi_label_emo)

r = train_model(model, embedding_matrix, y_vad, y_emo_lex, type_lex)
#results_reg, results_class = evaluate_model(model, y_train_)
pred_reg, pred_class_emo = model.predict(embedding_matrix)

r2 = r2_score(y_vad, pred_reg)
print('------------------------------------')
print('regression vad...')
print('mse', mean_squared_error(y_vad, pred_reg))
print('mae', mean_absolute_error(y_vad, pred_reg))
print('r2: ', r2)
#cf_matrix = confusion_matrix(labels=y_vad, predictions=pred_reg, num_classes=2)
#cf_matrix = np.array(cf_matrix)
#print(cf_matrix)

print('------------------------------------')
print('classification emo_lex...')
pred_class_emo = pred_class_emo.round()	
acc = accuracy_score(y_emo_lex, y_pred=pred_class_emo)
print('accuracy: ', acc)
cf_matrix = multilabel_confusion_matrix(y_true=y_emo_lex, y_pred=pred_class_emo)
cf_matrix = np.array(cf_matrix)
print(cf_matrix)


senti_embedding = merge_semantic_end_emotion_embeddings(model, embedding_matrix, type_matrix_emb)
print('Senti embeddings size PCA', np.shape(senti_embedding))
name_file = 'combined_class_reg'

save_senti_embeddings(senti_embedding, vocabulary_, vocabulary, name_file, type_matrix_emb)




