## Forecast target

Generally speaking, large e-commerce websites have their own category system, and the category planning of e-commerce plays an important role in the operation,
Our goal in this Demo is to use the pre trained BERT model to complete the task of classifying items on e-commerce websites, and to judge whether the item belongs to the Fashion industry by entering the title of the item.

## Model structure


For the model, we selected [HuggingFace Bert Base Cased pre training checkpoint](https://huggingface.co/bert-base-cased).
The model has 12 layers of Transformer Encoder, and the hidden layer dimension is 768, with a total of 110 million parameters.
Above the Bert layer are the Dropout layer, the Linear layer and the Sigmaid layer. The overall structure is as follows:

<img src="../../docs/images/title_to_fishion.PNG" alt="title_to_fish architecture" width="500">

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

After obtaining the data required for model training, you can run the training model script [train. sh] (text_classifier/train. sh) to train the model.

```
cd text_ classifier
bash train.sh
```

The following is the content of train.sh:

```
python -u train. py --name title_ to_ fashion \
--model bert-base-cased --num-labels 1 \
--train-file /your/working/path/title_ to_ fashion_ 500k/train. tsv \
--eval-file /your/working/path/title_ to_ fashion_ 500k/val. tsv \
--eval-steps 1000 \
--num-epochs 1 --lr 2e-5 --train-batch-size 32 --eval-batch-size 32 --gpu 0 \
--output-path ./output
```

In this script, you can customize the parameter configuration. The following describes the functions of the parameters:
+ name: assign the name of the model
+ model: assign the pre training model to be used
+ num-labels: the number of categories classified
+ train-file: the train file that generated in the data preparation phase
+ eval-file: the train file that generated in the data preparation phase
+ eval-steps: the number of steps to inference
+ num-epochs: the number of training epochs
+ lr: the learning rate in the training phase
+ train-batch-size: the batch size in the training stage
+ eval-batch-size: the batch size in the evaluate stage
+ gpu: assign the specified gpu
+ output-path: the saved path of the model.

## Model effect
The model training for an epoch has converged well. The following indicators are Accuracy, AUC, Precision, Recall, F1, etc. If we take Threshold=0.5, then

| Accuracy | AUC | Precision | Recall | F1 |
|:--------:|:---:|:---------:|:--------:|:--------:|
|0.9757|0.9943|0.9355|0.9389|0.9372|