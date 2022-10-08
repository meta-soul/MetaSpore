## Forecast target

Generally speaking, large e-commerce websites have their own category system, and the category planning of e-commerce plays an important role in the operation,
Our goal in this Demo is to use the pre trained BERT model to complete the task of classifying items on e-commerce websites, and to judge whether the item belongs to the Fashion industry by entering the title of the item.

## Model structure


For the model, we selected [HuggingFace Bert Base Cased pre training checkpoint](https://huggingface.co/bert-base-cased).
The model has 12 layers of Transformer Encoder, and the hidden layer dimension is 768, with a total of 110 million parameters.
Above the Bert layer are the Dropout layer, the Linear layer and the Sigmaid layer. The overall structure is as follows:
![title_to_fish architecture](docs/images/title_to_fish.PNG)

## Prepare data

We use [Amazon Review Data (2018)](https://nijianmo.github.io/amazon/)Datasets,
We selected 'Clothing, Shoes&Jewelry' as our positive sample, and other industries as our negative sample.
The original data set is large, and the training time is long. We randomly selected 500000 pieces of data,
These data can already meet our training objectives. Among them, train dataset accounts for 90%, dev dataset accounts for 5%, test dataset accounts for 5%, positive samples account for 20%, and negative samples account for 80%.
You can run the following script to generate data.

```
cd dataset
python -u gen_ data.py

```

## Model training

The training model script is in [train. sh] (text_classifier/train. sh), and you can configure parameters according to your environment.

```
cd text_ classifier
bash train.sh
```

## Model effect
The model training for an epoch has converged well. The following indicators are Accuracy, AUC, Precision, Recall, F1, etc. If we take Threshold=0.5, then

| Accuracy | AUC | Precision | Recall | F1 |
|:--------:|:---:|:---------:|:--------:|:--------:|
|0.9757|0.9943|0.9355|0.9389|0.9372|