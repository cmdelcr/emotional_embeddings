import os
import re

dict_emo_opposite = {'joy': 'sadness',
					 'trust': 'disgust',
					 'fear': 'anger',
					 'surprise': 'anticipation',
					 'sadness': 'joy',
					 'disgust': 'trust',
					 'anger': 'fear',
					 'anticipation': 'surprise'}

arr_emo = ['negative', 'positive']

vocab_emo = {}
with open('/home/carolina/corpora/lexicons/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt', 'r') as file:
	for line in file:
		rows = re.sub(r'\n', '', line).split('\t')
		if int(rows[2]) == 1:
			if rows[0] in vocab_emo:
				arr = vocab_emo[rows[0]]
				arr.append(rows[1])
			else:
				arr = [rows[1]]
			vocab_emo[rows[0]] = arr

counter_word = 0
with open('resources/vocabulary.txt', 'w') as file:
	for word in vocab_emo:
		file.write(word + '\n')
		counter_word += 1
print('Num words: ', counter_word, '\n-------------------------------------')


counter_true_emotion = 0
with open('resources/true_emotion.txt', 'w') as file:
	for word in vocab_emo:
		for emotion in vocab_emo[word]:
			if emotion not in arr_emo:
				file.write(word + '\t' + emotion + '\n')
				counter_true_emotion += 1
print('True emotions: ', counter_true_emotion)

counter_opposite_emotion = 0
with open('resources/opposite_emotion.txt', 'w') as file:
	for word in vocab_emo:
		for emotion in vocab_emo[word]:
			if emotion not in arr_emo:
				file.write(word + '\t' + dict_emo_opposite[emotion] + '\n')
				counter_opposite_emotion += 1
print('Opposite emotions: ', counter_true_emotion)

