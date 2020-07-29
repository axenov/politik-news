from transformers import BertTokenizer, BertForSequenceClassification
from nlp import load_dataset
from sklearn.metrics import accuracy_score
import torch

#Load dataset
dataset = load_dataset('de_politik_news.py', cache_dir='.de-politic-news')

#Tokenize test dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
encoded_test = dataset['test'].map(lambda examples: tokenizer(examples['text'], padding='max_length', truncation=True), batched=True)

#Process labels
label_dict = {'far-left':0, 'center-left':1, 'center':2, 'center-right':3, 'far-right':4}
encoded_test = encoded_test.map(lambda examples: {'labels': label_dict[examples['class']]})
encoded_test.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

#Initialize model
model = BertForSequenceClassification.from_pretrained('./model', num_labels=5)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

#Predict labels
class_test = encoded_test['labels']
model_test = []
for i in range(len(encoded_test)):
	prediction = model(**{k: v.unsqueeze(0).to(device) for k, v in encoded_test[i].items()})
	predicted = torch.argmax(prediction[1], dim=1).item()
	model_test.append(predicted)

#Calculate accuracy
accuracy = accuracy_score(class_test, model_test)
print(f'accuracy: {accuracy}')