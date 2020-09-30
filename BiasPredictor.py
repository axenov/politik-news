from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os
from nltk.corpus import stopwords
import numpy as np
import pickle

class biasPredictor():

	def __init__(self, method):

		#self.LABELS_DICT = {0:'far-left', 1:'center-left', 2:'center', 3:'center-right', 4:'far-right'}
		self.german_stop_words = stopwords.words('german')
		german_stop_words_temp = []
		for word in self.german_stop_words:
			german_stop_words_temp.append(word.capitalize())
		self.german_stop_words += german_stop_words_temp
		self.german_stop_words += ["`",'„','“','"','!',':','.','(','?',',',')',' ','-','_','*']

		self.method = method
		if method == 'bert':
			self.bert = BertForSequenceClassification.from_pretrained('models/BERT', num_labels=5).bert
			device = 'cuda' if torch.cuda.is_available() else 'cpu'
			self.bert.to(device)
			self.tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
			self.model = pickle.load(open('models/BERT/model.bin', 'rb'))

		elif method == 'tfidf':
			self.model = pickle.load(open('models/TFIDF/model.bin', 'rb'))
			self.vectorizer = pickle.load(open('models/TFIDF/vectorizer.bin', 'rb'))
			#self.importance = self.model.feature_importances_
			#self.inverse_vacabulary = dict([(id,word) for word,id in self.vectorizer.vocabulary_.items()])
		else:
			raise ValueError('Not existing method.')

	#NOT IMPLEMENTED
	def explain_rf_(self,text_vector):
		#num_words = min(10,round(len(text_vector.indices)/2))
		#ids = np.argsort(self.importance[text_vector.indices])[::-1][:num_words]
		#To concatenate adjacent tokens (need new vectorizer.transform() method to keep the tokens order)
		'''
		ids = list(np.sort(ids))
		ngrams = []
		temp_gram = []
		prev = ids[0]-1
		for i in ids:
			if (i - prev == 1) or temp_gram == []:
				temp_gram += [i]
			else:
				ngrams.append(temp_gram)
				temp_gram = [i]
			prev = i
		if (len(ngrams) == 0) or (ngrams[-1] != temp_gram):
			ngrams.append(temp_gram)
		'''
		#ngrams = ids
		#for ngram in ngrams:
		#	word = self.inverse_vacabulary[text_vector.indices[ngram]]
		#	if (word not in self.german_stop_words):
		#		yield word
		return ['']

	#Build list of n-grams
	def add_indexes_right_(self,word,indexes):
		new_word = word
		if new_word[-1]+1 < len(indexes):
			if '##' in self.tokenizer.decode([indexes[new_word[-1]+1]]):
				new_word.append(new_word[-1]+1)
				new_word = self.add_indexes_right_(new_word, indexes)
		return new_word
	def add_indexes_left_(self,word,indexes):
		new_word = word
		if ('##' in self.tokenizer.decode([indexes[new_word[0]]])) and (new_word[0]!=0):
			new_word = [new_word[0]-1]+new_word
			new_word = self.add_indexes_left_(new_word, indexes)
		return new_word
	def add_indexes_(self,word,indexes):
		new_word = self.add_indexes_right_(word, indexes)
		new_word = self.add_indexes_left_(new_word, indexes)
		return new_word

	def explain_bert_(self, attention, input_ids):
		num_words = min(10,round(len(input_ids[0][input_ids[0]!=0])/2))
		sum_attentions = attention[0][0,0,:,:]
		sum_attentions = sum_attentions.sum(dim = 0)
		ids = torch.argsort(sum_attentions,descending=True)[:num_words]
		ids = torch.sort(ids).values
		#Create N-grams
		ngrams = []
		temp_gram = []
		prev = ids.tolist()[0]-1
		for i in ids.tolist():
			if i - prev-1 <= 2 or temp_gram == []:
				temp_gram += list(range(prev+1,i+1))#.append(i)
			else:
				temp_gram = self.add_indexes_(temp_gram, input_ids[0])
				ngrams.append(torch.tensor(temp_gram))
				temp_gram = [i]
			prev = i
		temp_gram = self.add_indexes_(temp_gram, input_ids[0])
		ngrams.append(torch.tensor(temp_gram))
		#Build textual representation
		sequences = []
		for ngram in ngrams:
			ids2 = input_ids[0][ngram]
			ids2 = ids2[(ids2!=2) & (ids2!=3) & (ids2!=4)]
			if ids2.shape[0] != 0:
				sequences.append((self.tokenizer.decode(ids2)))
		sequences = list(set(sequences))
		#remove stop words
		for sec in sequences:
			if (sec not in self.german_stop_words):
				yield(sec)


	def predict(self, explain = False, **kwargs):

		if 'text' in kwargs:
			text = kwargs['text']
		elif 'file_path' in kwargs:
			with open(kwargs['file_path']) as f:
				text = f.read()
		else:
			raise ValueError('No input provided')

		if self.method == 'bert':
			tokens = self.tokenizer(text, padding='max_length', truncation=True, return_tensors="pt")
			prediction = self.bert(**tokens, output_attentions=True)
			cl = self.model.predict(prediction[1].cpu().detach().numpy())[0]
			#cl = self.LABELS_DICT[torch.argmax(prediction[0], dim=1).item()]
			if explain:
				explanation = list(self.explain_bert_(prediction[2],tokens['input_ids']))

		elif self.method == 'tfidf':
			text_vector = self.vectorizer.transform([text])
			cl = self.model.predict(text_vector)[0]
			if explain:
				explanation = list(self.explain_rf_(text_vector))

		if explain:
			return cl, explanation
		else:
			return cl

