from transformers import BertTokenizer, BertForSequenceClassification
import torch
import argparse
import os

class biasPredictor():

	def __init__(self, model_path):

		self.LABELS_DICT = {0:'far-left', 1:'center-left', 2:'center', 3:'center-right', 4:'far-right'}
		self.model = BertForSequenceClassification.from_pretrained(model_path, num_labels=5)
		device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.model.to(device)
		self.tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')

	def __run_model__(self, tokens):
		return self.model(**tokens, output_attentions=True)

	def predict(self, explain = False, **kwargs):

		if 'text' in kwargs:
			tokens = self.tokenizer(kwargs['text'], padding='max_length', truncation=True, return_tensors="pt")
		elif 'file_path' in kwargs:
			with open(kwargs['file_path']) as f:
				tokens = self.tokenizer(f.read(), padding='max_length', truncation=True, return_tensors="pt")
		else:
			raise ValueError('No input provided')

		prediction = self.__run_model__(tokens)
		cl = self.LABELS_DICT[torch.argmax(prediction[0], dim=1).item()]

		if explain:
			explanation = self.explain(prediction[1])
		else:
			explanation = None

		return cl, explanation

	def explain(self, attention):
		return ['Trump']
