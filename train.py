from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from nlp import load_dataset
from tqdm import tqdm
import torch
import argparse

# Construct the argument parser
ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument("-batch_size", "--batch_size", required=True,
   help="Batch size")
ap.add_argument("-num_epochs", "--num_epochs", required=True,
   help="Number of epochs")
args = vars(ap.parse_args())

BATCH_SIZE = args['batch_size']
NUM_EPOCHS = args['num_epochs']

#Load dataset
dataset = load_dataset('de_politik_news.py', cache_dir='.de-politic-news')

#Tokenize test and validation datasets
tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
encoded_train = dataset['train'].map(lambda examples: tokenizer(examples['text'], padding='max_length', truncation=True), batched=True)
encoded_valid = dataset['validation'].map(lambda examples: tokenizer(examples['text'], padding='max_length', truncation=True), batched=True)

#Process labels
label_dict = {'far-left':0, 'center-left':1, 'center':2, 'center-right':3, 'far-right':4}
encoded_train = encoded_train.map(lambda examples: {'labels': label_dict[examples['class']]})
encoded_valid = encoded_valid.map(lambda examples: {'labels': label_dict[examples['class']]})

#Load data
encoded_train.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
dataloader = torch.utils.data.DataLoader(encoded_train, batch_size=BATCH_SIZE)

#Initialize model
model = BertForSequenceClassification.from_pretrained('bert-base-german-cased', num_labels=5)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.train().to(device)

optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)


for epoch in range(NUM_EPOCHS):
    for i, batch in enumerate(tqdm(dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % 100 == 0:
            model.save_pretrained('./model')
            print(f"loss: {loss}")

model.save_pretrained('./model')