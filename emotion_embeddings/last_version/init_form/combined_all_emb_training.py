import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, r2_score, mean_absolute_error
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
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.backend import epsilon
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


class Custom_Weighted_CE_Loss(tf.keras.losses.Loss):
	def __init__(self):
		super().__init__()

	def __init__(self, class_weights):
		super().__init__()
		self.class_weights = class_weights

	def call(self, y_true, y_pred):		
		log_y_pred = tf.math.log(y_pred)# + 0.0000001)
		y_true = tf.cast(y_true, dtype=tf.float32)
		elements = tf.math.multiply_no_nan(x=log_y_pred, y=y_true)# + 0.0000001)
		#elements = tf.math.multiply_no_nan(x=elements, y=self.class_weights)
		loss = -tf.reduce_mean(tf.reduce_sum(elements,axis=1))

		return loss

def def_weigths(arr_class_counter):
	global class_weights
	positive_weights = {}
	negative_weights = {}

	for idx in range(len(arr_class_counter)):
		positive_weights[idx] = np.sum(arr_class_counter) / (2 * arr_class_counter[idx])
		negative_weights[idx] = np.sum(arr_class_counter) / (2 * (np.sum(arr_class_counter) - arr_class_counter[idx]))
		
	class_weights['positive_weights'] = positive_weights
	class_weights['negative_weights'] = negative_weights


def custom_loss(y_true, y_logit):
    '''
    Multi-label cross-entropy
    * Required "Wp", "Wn" as positive & negative class-weights
    y_true: true value
    y_logit: predicted value
    '''
    loss = float(0)
    y_true = tf.cast(y_true, dtype=tf.float32)
    
    for i, key in enumerate(class_weights['positive_weights'].keys()):
        first_term = class_weights['positive_weights'][key] * y_true[i] * tf.math.log(y_logit[i] + epsilon())
        second_term = class_weights['negative_weights'][key] * (1 - y_true[i]) * tf.math.log(1 - y_logit[i] + epsilon())
        loss -= (first_term + second_term)
    return loss

def create_model(input_shape, output_classification_size):
	input_ = Input(shape=(input_shape,), name='input_layer')
	hidden_shared_layer = Dense(100, activation='relu', name='hidden_shared_layer')#, kernel_initializer='he_normal') 
	x1 = hidden_shared_layer(input_)

	# layers classification
	hidden_layer = Dense(50, activation='relu', name='first_class_layer')#, 
		#kernel_regularizer=regularizers.l2(0.0001), bias_regularizer=regularizers.l2(0.0001)) 
	x2 = hidden_layer(x1)
	hidden_layer1 = Dense(20, activation='relu', name='second_class_layer')#, 
		#kernel_regularizer=regularizers.l2(0.0001), bias_regularizer=regularizers.l2(0.0001)) 
	x2 = hidden_layer1(x2)

	output_regression = Dense(3, activation='linear', name='output_reg')(x1)
	output_classification = Dense(output_classification_size, activation='sigmoid', name='output_class')(x2)

	model = Model(inputs=[input_],outputs=[output_regression, output_classification])


	return model

def compile_model(model, arr_class_counter, multi_label):
	def_weigths(arr_class_counter)
	model.compile(
			loss=['mean_squared_error', 
					'categorical_crossentropy' 
					#Custom_Weighted_CE_Loss(np.asarray(list(def_class_weight(arr_class_counter).values()), dtype='float32'))
					if not multi_label else 
					BinaryCrossentropy()],
					#{'multi_o/p': custom_loss}
					#],
			#Custom_Weighted_CE_Loss(def_class_weight(arr_class_counter))], # 'categorical_crossentropy'
			optimizer='adam',#Adam(learning_rate=0.001),#'adam',
			metrics=[tf.keras.metrics.RootMeanSquaredError(), 'accuracy']
	)
	plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

	return model

def def_class_weight(arr_class_counter):
	arr_class = {}
	for idx in range(len(arr_class_counter)):
		#n_samples / (n_classes * n_samplesj)
		arr_class[idx] = np.sum(arr_class_counter) / (len(arr_class_counter) * arr_class_counter[idx])

	print('class_weight: ', arr_class)
	#exit()
	return arr_class

def train_model(model, x_train, y_vad, y_sub, type_lex):
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
		batch_size=256, 
		epochs=50,#150 
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
#type_lex = 'sub_clues'
#dict_sub, arr_class_counter = read_subjectivity_clues()
type_lex = 'emo_lex'
multi_label = True
dict_sub, arr_class_counter = read_emo_lex_file(only_emotions=True, multi_label=multi_label)

word2vec = read_embeddings(emb_type)
word2idx, vocabulary = getting_lemmas(emb_type, dict_vad, dict_sub, word2vec)

print('-----------------------------------------')
print('***Filling pre-trained embeddings...')
embedding_matrix, vocabulary_, y_vad, y_sub = filling_embeddings(
			word2idx, word2vec, vocabulary, dict_vad, dict_sub, emb_type, type_matrix_emb, type_lex)
print('size_embeddings: ', np.shape(embedding_matrix))


#minmax_scale = preprocessing.MinMaxScaler(feature_range=(-1, 1))
#y_vad = minmax_scale.fit_transform(y_vad)


print('--------------------------')
print('Creating the model...')
model = create_model(len(embedding_matrix[0]), len(y_sub[0]))
model = compile_model(model, arr_class_counter, multi_label)

r = train_model(model, embedding_matrix, y_vad, y_sub, type_lex)
#results_reg, results_class = evaluate_model(model, y_train_)
pred_reg, pred_class = model.predict(embedding_matrix)

r2 = r2_score(y_vad, pred_reg)
print('------------------------------------')
print('mae', mean_absolute_error(y_vad, pred_reg))
print('r2: ', r2)

pred_class = pred_class.round()	
acc = accuracy_score(y_sub, y_pred=pred_class)
print('accuracy: ', acc)


senti_embedding = merge_semantic_end_emotion_embeddings(model, embedding_matrix, type_matrix_emb)
print('Senti embeddings size PCA', np.shape(senti_embedding))
name_file = 'combined_class_reg'

save_senti_embeddings(senti_embedding, vocabulary_, vocabulary, name_file, type_matrix_emb)




