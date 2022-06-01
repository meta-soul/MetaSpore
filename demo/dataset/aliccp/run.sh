#!/bin/sh
# small dataset
wget https://paddlerec.bj.bcebos.com/esmm/traindata_10w.csv
wget https://paddlerec.bj.bcebos.com/esmm/testdata_10w.csv

aws s3 cp traindata_10w.csv s3://dmetasoul-bucket/demo/aliccp/
aws s3 cp testdata_10w.csv s3://dmetasoul-bucket/demo/aliccp/

# big dataset
wget https://paddlerec.bj.bcebos.com/datasets/aitm/ctr_cvr.train
wget https://paddlerec.bj.bcebos.com/datasets/aitm/ctr_cvr.test

mv ctr_cvr.train traindata_4000w.csv
mv ctr_cvr.test testdata_4000w.csv

aws s3 cp traindata_4000w.csv s3://dmetasoul-bucket/demo/aliccp/
aws s3 cp testdata_4000w.csv s3://dmetasoul-bucket/demo/aliccp/
