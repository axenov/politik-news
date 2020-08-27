# Political Bias Classification of German Media
This project is the first attemp to do Political Bias classification of German news.

We clawled out data from various German news sites using [news-please library](https://github.com/fhamborg/news-please). After that we manually cleaned the data and labeled it using [Medienkompass](https://medienkompass.org/). Data is organised as [HuggingFace nlp library](https://github.com/huggingface/nlp) dataset. 

Due to the copyright issues we can not publish the data, but soon we will published the list of urls you can use to build this dataset by your own.

Our system uses German BERT from [HuggingFace Transformers library](https://github.com/huggingface/transformers) as the pre-trained model to fine-tune.


To train model run:
```bash
python train.py -data_folder="data" model_folder="model" -batch_size=8 -num_epochs=2
```

To test model run:
```bash
python test.py -data_folder="data" model_folder="model"
```

### Effect of Covid-19 on German news

<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/axenov/politik-news/master/docs/imgs/corona.png" width="400"/>
    <br>
<p>

The web demo will be released soon.
