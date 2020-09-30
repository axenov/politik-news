# Political Bias Classification of German Media
This project is the first attemp to do Political Bias classification of German news.

We clawled out data from various German news sites using [news-please library](https://github.com/fhamborg/news-please). After that we manually cleaned the data and labeled it using [Medienkompass](https://medienkompass.org/). Data is organised as [HuggingFace nlp library](https://github.com/huggingface/nlp) dataset. 

Due to the copyright issues we can not publish the data, but provided the list of urls you can use to build this dataset by your own.
To download all the data run:

```python
NewsPlease.from_file('data/urls.txt')
```

Then run (under development):
```python
python preprocess.ty -data_folder='path/to/your/downloaded/data'
```

Our system is based on Random Forest algorithm applied to BERT and TF-IDF features.
Our implementation of BERT uses [HuggingFace Transformers library](https://github.com/huggingface/transformers).


To fine-tune pretrained BERT model run:
```bash
python train.py -data_folder="data" model_folder="models/BERT" -batch_size=8 -num_epochs=2
```

To test this model on the test data run:
```bash
python test.py -data_folder="data" model_folder="models/BERT"
```

To process a single arbitary text and get the list of the most important words run:
```bash
python predict.py -file_path="text_sample.txt" -method="tfidf" -explain=true
```

or call in python:
```python
from BiasPredictor import biasPredictor
predictor = biasPredictor("path/to/the/model")
prediction = predictor.predict(text = "Trump ist Schlecht", explain=True)
print(prediction)
#outut: ('center-rignt',[Trump])
```


### t-SNE on SVD of BOW representation of the dataset

<p align="center">
    <br>
    <img src="https://github.com/axenov/politik-news/blob/master/docs/imgs/SVD.png" width="700"/>
    <br>
<p>

### Effect of Covid-19 on German news

<p align="center">
    <br>
    <img src="https://github.com/axenov/politik-news/blob/master/docs/imgs/covid.png" width="700"/>
    <br>
<p>

The web demo will be released soon.
