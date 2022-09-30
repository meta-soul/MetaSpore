# Text Classification

The objective of this Demo is using pre-train BERT model to classify items of ecommerce website.
The input is item's title and the output tells us whether this item belongs to Fashion field or not.
If you are Chinese developer, you may like to visit our [CN Doc](README-CN.md).


## 1. Preparation of dataset
We use [Amazon Review Data(2018)](https://nijianmo.github.io/amazon/) and sample 500K data using [gen_data.py](dataset/gen_data.py) to train our model.
```
cd dataset
python -u gen_data.py
```

## 2. Training
The entrypoint of training code is [train.sh](text_classifier/train.sh). You could configure it according to your environment.
```
cd text_classifier
bash train.sh
```

## 3. Prediction
The entrypoint of prediction code is main 
```
cd text_classifier
bash train.sh
```

## 4. Launch recommend online service
You could run online service entry point (MovielensRecommendApplication.java) and test it now.
For example: `curl http://localhost:8080/user/10` to get recommended movies for user whose userId is equal to 10.

