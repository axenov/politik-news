import os
import random
import pandas as pd
import json
import jsonlines
from nlp import load_dataset
import argparse

def set_class(bias):
    if bias < 3:
        return 'far-left'
    elif 3 <= bias < 3.7:
        return 'center-left'
    elif 3.7 <= bias < 4.3:
        return 'center'
    elif 4.3 <= bias < 5:
        return 'center-right'
    if bias >= 5:
        return 'far-right'

#Construct the argument parser
ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument("-min_year", "--min_year", 
   help="Minimal publishing year of the article")

args = vars(ap.parse_args())

YEAR = args['min_year']

basepath = 'data/'
outputpath = 'data_preprocessed/'
labelpath = 'labels/'

#Load labels
labels = pd.read_csv(labelpath + 'labels.csv', index_col=0).to_dict()
liberal_right_labels = pd.read_csv(labelpath + 'liberal-right-labels.csv', index_col=0).to_dict()
liberal_left_labels = pd.read_csv(labelpath + 'liberal-left-labels.csv', index_col=0).to_dict()
conservative_right_labels = pd.read_csv(labelpath + 'conservative-right-labels.csv', index_col=0).to_dict()
conservative_left_labels = pd.read_csv(labelpath + 'conservative-left-labels.csv', index_col=0).to_dict()
site_list = pd.read_csv(labelpath + 'site_list.csv', index_col=1).to_dict()['media']

#Process json files
json_entries = []
for entry in os.listdir(basepath):
    folderpath = os.path.join(basepath, entry)
    if os.path.isdir(folderpath):
        for textfile in os.listdir(folderpath):
            filepath = os.path.join(folderpath, textfile)
            if os.path.isfile(filepath):
                with open(filepath) as json_file:
                    data = json.load(json_file)
                    site = site_list[data['source_domain']]
                    
                    data.pop('date_download')
                    data.pop('date_modify')
                    data.pop('filename')
                    data.pop('image_url')
                    data.pop('language')
                    data.pop('title_page')
                    data.pop('title_rss')
                    data.pop('localpath')
                    
                    data['authors'] = '; '.join(data['authors'])
                    data['bias'] = labels['bias'][site]
                    data['quality'] = labels['quality'][site]
                    data['class'] = set_class(data['bias'])
                    
                    data['conservative_left_bias'] = conservative_left_labels['bias'][site]
                    data['conservative_left_quality'] = conservative_left_labels['quality'][site]
                    data['conservative_left_class'] = set_class(data['conservative_left_bias'])
                    
                    data['liberal_left_bias'] = liberal_left_labels['bias'][site]
                    data['liberal_left_quality'] = liberal_left_labels['quality'][site] 
                    data['liberal_left_class'] = set_class(data['liberal_left_bias'])
            
                    data['liberal_right_bias'] = liberal_right_labels['bias'][site]
                    data['liberal_right_quality'] = liberal_right_labels['quality'][site]
                    data['liberal_right_class'] = set_class(data['liberal_right_bias'])
                    
                    data['conservative_right_bias'] = conservative_right_labels['bias'][site]
                    data['conservative_right_quality'] = conservative_right_labels['quality'][site]   
                    data['conservative_right_class'] = set_class(data['conservative_right_bias'])

                    json_entries.append(data)


#Filter old articles
if YEAR:
    YEAR = int(YEAR)
    json_entries_new = []
    class_dict_new = {'far-left':0, 'far-right':0,'center-left':0,'center-right':0,'center':0}

    for entry in json_entries:
        if (entry['date_publish'] != None):
            if (int(entry['date_publish'].split('-')[0]) >= YEAR):
                json_entries_new.append(entry)
        else:
            json_entries_new.append(entry)
else:
	json_entries_new = json_entries


#Save data
random.seed(42)
random.shuffle(json_entries_new)

test_num_new = int(len(json_entries_new)*0.1)
train_entries_new = json_entries_new[:-2*test_num_new]
validation_entries_new = json_entries_new[-2*test_num_new:-test_num_new]
test_entries_new = json_entries_new[-test_num_new:]

with jsonlines.open(outputpath + 'train.jsonl', 'w') as writer_train:
    writer_train.write_all(train_entries_new)
with jsonlines.open(outputpath + 'validation.jsonl', 'w') as writer_valid:
    writer_valid.write_all(validation_entries_new)
with jsonlines.open(outputpath + 'test.jsonl', 'w') as writer_test:
    writer_test.write_all(test_entries_new)
