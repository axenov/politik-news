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

Our system uses German BERT from [HuggingFace Transformers library](https://github.com/huggingface/transformers) as the pre-trained model to fine-tune.


To train model run:
```bash
python train.py -data_folder="data" model_folder="model" -batch_size=8 -num_epochs=2
```

To test model run:
```bash
python test.py -data_folder="data" model_folder="model"
```

To process a single arbitary text and get the list of the most important words run:
```bash
python predict.py -file_path="text_sample.txt" -model_folder="path/to/the/model" -explain=true
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
