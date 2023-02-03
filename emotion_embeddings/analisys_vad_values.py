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

def get_bins(x, label):
	q25, q75 = np.percentile(x, [25, 75])
	bin_width = 2 * (q75 - q25) * len(x) ** (-1/3)
	bins = round((x.max() - x.min()) / bin_width)
	print("Freedman–Diaconis number of bins in " + label + ":", bins)

	return bins

plot_fig = False
show_values_between_vectors = False
list_word = []
list_vad = []
df = pd.read_csv('/home/carolina/corpora/lexicons/vad_lexicons/NRC-VAD-Lexicon/NRC-VAD-Lexicon.txt', 
			keep_default_na=False, header=None, sep='\t')

for index, row in df.iterrows(): #V, A, D
	list_word.append(str(row[0]).lower())
	list_vad.append([float(row[1]), float(row[2]), float(row[3])])


list_vad = np.array(list_vad)
valence = list_vad[:, 0]
arousal = list_vad[:, 1]
dominance = list_vad[:, 2]




corr = np.corrcoef(valence, arousal)
print(corr)
corr = np.corrcoef(valence, dominance)
print(corr)
corr = np.corrcoef(arousal, dominance)
print(corr)

corr = np.corrcoef(list_vad[:, 0], list_vad[:, 1])
print(corr)



if show_values_between_vectors:

	valence = list_vad[0:50, 0]
	arousal = list_vad[0:50, 1]
	dominance = list_vad[0:50, 2]

	fig, (axs1, axs2, axs3) = plt.subplots(1, 3)
	axs1.scatter(valence, arousal)
	axs1.set(xlabel='valence', ylabel='arousal')
	axs1.grid(color = 'gray', linestyle = '--', linewidth = 0.5)

	axs2.scatter(valence, dominance)
	axs2.set(xlabel='valence', ylabel='dominance')
	axs2.grid(color = 'gray', linestyle = '--', linewidth = 0.5)

	axs3.scatter(arousal, dominance)
	axs3.set(xlabel='arousal', ylabel='dominance')
	axs3.grid(color = 'gray', linestyle = '--', linewidth = 0.5)

	plt.show()


if plot_fig:

	fig, (axs1, axs2, axs3) = plt.subplots(1, 3)
	axs1.hist(valence, bins=get_bins(valence, 'valence'))
	axs1.set(xlabel='Valence', ylabel='Number of words')
	axs1.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
	axs1.set_ylim([0, 1200])

	axs2.hist(arousal, bins=get_bins(arousal, 'arousal'))
	axs2.set(xlabel='Arousal')
	axs2.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
	axs2.set_ylim([0, 1200])

	axs3.hist(dominance, bins=get_bins(dominance, 'dominance'))
	axs3.set(xlabel='Dominance')
	axs3.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
	axs3.set_ylim([0, 1200])

	plt.show()