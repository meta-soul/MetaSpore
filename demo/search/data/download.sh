#!/bin/bash

echo "Download DuReader_retrieval dataset"
wget -nv https://dataset-bj.cdn.bcebos.com/qianyan/dureader-retrieval-baseline-dataset.tar.gz
tar -zxvf dureader-retrieval-baseline-dataset.tar.gz
mv dureader-retrieval-baseline-dataset/dev ./
mv dureader-retrieval-baseline-dataset/passage-collection ./
mv dureader-retrieval-baseline-dataset/License.docx ./
mv dureader-retrieval-baseline-dataset/readme.md ./
mkdir ./test && mv dureader-retrieval-baseline-dataset/test.* ./test
rm dureader-retrieval-baseline-dataset.tar.gz
rm -rf dureader-retrieval-baseline-dataset

wget -nv https://dataset-bj.cdn.bcebos.com/qianyan/dureader_retrieval-data.v0.tar.gz
tar xvf dureader_retrieval-data.v0.tar.gz dureader_retrieval-data/train.json
mkdir ./train && mv dureader_retrieval-data/train.json ./train
rm dureader_retrieval-data.v0.tar.gz
rm -rf dureader_retrieval-data
