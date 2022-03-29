# MetaSpore

## Data preprocessing(fg_movielens.py)
After we read the data from the original database, we process the data to be used by the ranking model
1.Manually initialize a path.
export MY_S3_BUCKET='your item_cf bucket path'
envsubst <input.yaml> output.yaml
2.run the code

python fg_movielens.py -conf fg.yaml


Data preprocessed will be saved to 'item_summary_mongo_dataset_out_path','item_fearture_mongo_dataset_out_path'and'user_mongo_dataset_out_path'.
 
 
## Retrieval algorithm 

In this section ,In this section, we use ItemCF,swing and simpleX to get items. The code runs with the following command:



#### item_cf
1.Manually initialize a path.
export MY_S3_BUCKET='your item_cf bucket path'
envsubst <input.yaml> output.yaml

#### !!!Notice:input.yaml can't be same to output.yaml.

2.run the code

python item_cf.py -conf item_cf.yaml

3.Write model running the output data into mongo database(.dump/write_mongo.py)
python write_mongo.py

#### 2.swing

1.Manually initialize a path.
export MY_S3_BUCKET='your swing bucket path'
envsubst <input.yaml> output.yaml
### !!!Notice:input.yaml can't be same to output.yaml.

2.run the code

python swing.py -conf swing.yaml
Notice: input.yaml can't ba same to output.yaml

#### 3.simpleX  
1.Manually initialize a path.
export MY_S3_BUCKET='your swing bucket path'
envsubst <input.yaml> output.yaml

### !!!Notice:input.yaml can't be same to output.yaml.
2.run the code

python swing.py -conf swing.yaml

# Ranking model

ranking model include widedeep and light_gbm presently.
### 1.Manually initialize a path.

export MY_S3_BUCKET='your bucket path'
envsubst <input.yaml> output.yaml

## 2.run the code
python 'model.py' -conf 'model.yaml'

## Tuner
#### 1.Manually initialize a path.
export MY_S3_BUCKET='your bucket path
envsubst <input.yaml> output.yaml

#### 2.run the code

