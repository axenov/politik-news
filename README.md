# Political Bias Classification of German Media
This project is the first attempt to do Political Bias classification of German news.

We crawled out data from various German news sites using [news-please library](https://github.com/fhamborg/news-please). After that, we manually cleaned the data and labeled it using [Medienkompass](https://medienkompass.org/). Then the dataset was preprocessed using [HuggingFace NLP library](https://github.com/huggingface/nlp). 

Due to copyright issues, we can not publish the data, but we provided the list of URLs you can use to build this dataset on your own.
To download all the data run:

```python
NewsPlease.from_file('urls/urls.txt')
```
Then run the preprocessing script:
```python
python preprocess.py -data_folder='path/to/your/downloaded/data'
```
We evaluated several classification models on the dataset, using Bag-of-Words, TF-IDF, and BERT features. For reproduction the former two, run *BOW_baseline.ipynb* and *TFIDF_baseline.ipynb* notebooks. To train BERT-based models you need to fine-tune [HuggingFace](https://github.com/huggingface/transformers) implementation of German BERT.
```bash
python train.py -data_folder="data" model_folder="models/BERT" -batch_size=8 -num_epochs=2
```
After that run *BERT_baseline.ipynb* notebook.

Using our two based models for TF-IDF and BERT features, we implemented the demo system that can predict the political bias of a single arbitary text and generate the list of the words that pushes the system to make the decision. The models can be download from [here](https://drive.google.com/file/d/1dUu9sYEXU0C5CHzGPocoDPCCbVrQD1Q8/view?usp=sharing).
To use the system run:
```bash
python predict.py -file_path="text_sample.txt" -method="tfidf" -explain=False
```
or call in python:
```python
from BiasPredictor import biasPredictor
predictor = biasPredictor("bert")
prediction = predictor.predict(text = "Ein politischer Text", explain=True)
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
