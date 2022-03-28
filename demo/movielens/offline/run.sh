 #!/bin/sh
# config
fg_=0
match=0
rank=1

# feature genereation
if [ $fg_ = 1 ]
then
    python fg_movielens.py --conf fg.yaml
fi

# retrival
if [ $match = 1 ]
then 
    cp ../../python.zip .
    aws s3 cp  --recursive ./schema/simplex/ s3://dmetasoul-bucket/demo/movielens/schema/simplex/
    python simplex.py --conf simplex.yaml
fi

# rank
if [ $rank = 1 ]
then
    cp ../../python.zip .
    aws s3 cp  --recursive ./schema/widedeep/ s3://dmetasoul-bucket/demo/movielens/schema/widedeep/
    python widedeep.py --conf widedeep.yaml
    python lgbm_model_train.py --conf lgbm.yaml
fi

