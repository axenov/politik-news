from transformers import BertTokenizer, BertForSequenceClassification
from nlp import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch
import argparse
import os

# Construct the argument parser
ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument("-data_folder", "--data_folder", required=True,
   help="Path to the dataset")
ap.add_argument("-model_folder", "--model_folder", required=True,
   help="Path to the model")

args = vars(ap.parse_args())


DATA_FOLDER = args['data_folder']
MODEL_FOLDER = args['model_folder']
#Load dataset
#dataset = load_dataset(os.path.join(DATA_FOLDER, 'de_politik_news.py'), cache_dir=os.path.join(DATA_FOLDER, '.de-politic-news'))
dataset = load_dataset('de_politik_news.py', cache_dir=DATA_FOLDER)
#Tokenize test dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
encoded_test = dataset['test'].map(lambda examples: tokenizer(examples['text'], padding='max_length', truncation=True), batched=True)

#Process labels
label_dict = {'far-left':0, 'center-left':1, 'center':2, 'center-right':3, 'far-right':4}
encoded_test = encoded_test.map(lambda examples: {'labels': label_dict[examples['class']]})

class_test = encoded_test['labels']
encoded_test.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask'])#, 'labels'])

#Initialize model
model = BertForSequenceClassification.from_pretrained(MODEL_FOLDER, num_labels=5)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

#Predict labels
model_test = []
for i in range(len(encoded_test)):
	prediction = model(**{k: v.unsqueeze(0).to(device) for k, v in encoded_test[i].items()})
	predicted = torch.argmax(prediction[0], dim=1).item()
	model_test.append(predicted)

#Calculate accuracy
accuracy = accuracy_score(class_test, model_test)
f1_micro = f1_score(class_test, model_test, average = 'micro')
f1_macro = f1_score(class_test, model_test, average = 'macro')
report = classification_report(class_test, model_test)
print(f'accuracy: {accuracy}')
print(f'F1-micro: {f1_micro}')
print(f'F1-macro: {f1_macro}')
print(f'Report: {report}')
