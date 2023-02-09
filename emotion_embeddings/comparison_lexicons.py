import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import r2_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from gensim import models

import matplotlib.pyplot as plt
import scipy.stats as st

from util import *
import settings
df_vad = pd.read_csv('/home/carolina/corpora/lexicons/vad_lexicons/NRC-VAD-Lexicon/NRC-VAD-Lexicon.txt', 
			keep_default_na=False, header=None, sep='\t')

dict_vad = {}
for index, row in df_vad.iterrows(): #V, A, D
	dict_vad[str(row[0]).lower()] = [float(row[1]), float(row[2]), float(row[3])]
print('size_vad: ', len(dict_vad))

#(anger, fear, anticipation, trust, surprise, sadness, joy, or disgust) or one of two polarities (negative or positive)
df_emo_lex = pd.read_csv('/home/carolina/corpora/lexicons/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt', 
			keep_default_na=False, header=None, sep='\t')

arr_emotions = ['anger', 'fear', 'anticipation', 'trust', 'surprise', 'sadness', 'joy', 'disgust', 'negative', 'positive']
dict_emo_lex = {}
for index, row in df_emo_lex.iterrows():
	if str(row[0]) in dict_emo_lex:
		arr_emo_lex = dict_emo_lex[str(row[0])]
	else:
		arr_emo_lex = np.zeros(10)
	arr_emo_lex[arr_emotions.index(str(row[1]))] = int(row[2])
	dict_emo_lex[str(row[0])] = arr_emo_lex
print('size_emo_lex: ', len(dict_emo_lex))


df_int_emo = pd.read_csv('/home/carolina/corpora/lexicons/NRC-Emotion-Intensity-Lexicon/NRC-Emotion-Intensity-Lexicon-v1.txt', 
			keep_default_na=False, header=None, sep='\t')

dict_int_emo = {}
for index, row in df_int_emo.iterrows():
	if str(row[0]) in dict_int_emo:
		arr_int_emo = dict_int_emo[str(row[0])]
	else:
		arr_int_emo = np.zeros(10)
	arr_int_emo[arr_emotions.index(str(row[1]))] = float(row[2])
	dict_int_emo[str(row[0])] = arr_int_emo
print('size_int_emo: ', len(dict_int_emo))


list_vad = list(dict_vad.keys())
list_emo_lex = list(dict_emo_lex.keys())
list_int_emo = list(dict_int_emo.keys())

set_vad = set(list_vad)
set_emo_lex = set(list_emo_lex)
set_int_emo = set(list_int_emo)
set_rest = set_emo_lex - set_vad

print(len(set_vad.intersection(set_emo_lex)))
print(len(set_vad.intersection(set_int_emo)))
#print(len(set_vad.difference(set_emo_lex)))
