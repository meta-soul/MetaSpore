## [中文文档](README-CN.md)

# 1. Introduction

**Information retrieval** is a basic problem in many scenarios such as general/domain search, question and answer, etc. Once retrieval technology builds a semantic bridge between query and item, many application scenarios related to information retrieval will benefit. In real application scenarios, the scale of data to be processed by retrieval system is often very big. For example, the product database of e-commerce search reaches one billion levels, and general search needs to process ten trillion levels of web page data. Therefore, we should trade-off between accuracy and performance, the search pipeline often be divided into two-stage retrieval and reranking. In the **retrieval stage**, it is need to quickly and efficiently search out hundreds of thousands of candidate item sets from a large-scale item database. In the **reranking stage**, richer features and more complex models can be used to refine the retrieval candidate item set.

Here we focus on problems such as model offline optimization and online inference engineering architecture in the retrieval systems. In terms of model offline optimization, we will implement the SOTA **dense vector retrieval** and **deep semantic ranking** methods based on the open source [dataset](https://aistudio.baidu.com/aistudio/competition/detail/157/0/introduction). In terms of model online inference, we will start from the offline distillation, quantization, and combine [MetaSpore Serving](https://github.com/meta-soul/MetaSpore) to improve the model online inference performance.

# 2. Dataset

Both our dense vector retrieval and deep semantic ranking models are optimized based on a real open dataset in industry—[DuReaderRetrieval](https://aistudio.baidu.com/aistudio/competition/detail/157/0/introduction). Mainly considering that this dataset has several advantages:

- From Baidu search online log
- A high-quality and large-scale Chinese text retrieval dataset
- The quality of the evaluation set be improved by annotators

There are some insights about this dataset through the data analysis:

- Query is short text, less then 10 chars
- Passage database is big, about 8M examples
- Only positive example in the train data, developers need dig negatives by themself
- There are many false negatives in the train data, e.g, there are 2.57 positives for each query in training data, but 4.93 positives in evaluation data

We will consider the characteristics of the datasets to design model optimization strategies, and deeply explore the techniques of negative sample mining and hard negative sample denoising to overcome the shortcomings of the dataset.

*Dataset Statistics:*

|       | num of query | num of passage | passages for query | query length | passage length |
| ----- | ------------ | -------------- | ------------------ | ------------ | -------------- |
| Train | 86395        | 222395         | 2.57               | 9.51         | 362.6          |
| Valid | 2000         | 9863           | 4.93               | 9.28         | 387.64         |

The dataset can be obtained by the following command:

```bash
sh data/download.sh
```

# 3. Baseline

We use the [RocketQA Baseline](https://github.com/PaddlePaddle/RocketQA/tree/main/research/DuReader-Retrieval-Baseline) released by Baidu as the baseline model for comparison. The evaluation result of a baseline model on the **validation set**:

| Model |  MRR@10 | recall@1 | recall@50 |
| --- | --- | --- | --- |
| dual-encoder (retrieval) | 60.45 | 49.75 | 91.75|
| cross-encoder (re-ranking) | 72.84 | 64.10 | 91.75|

> Recall@50 is more suitable for evaluating the firststage retrievers, while MRR@10 and Recall@1 are more suitable for assessing the second-stage re-rankers.

We also checked the performance of a general semantic representation [model](https://huggingface.co/DMetaSoul/sbert-chinese-general-v2) that not fine-tuned on the retrieval task and found that the performance was poor (MRR@ 10=7.78%, recall@1=4.3%, recall@50=29.65%), the cross-domain transfer ability of the semantic representation model is weak, and it is necessary to train on the retrieval task.

In addition, DuReaderRetrieval [Dataset paper](https://arxiv.org/pdf/2203.10232.pdf) also gives the evaluation results of some other baseline models on the **test set** (not open) (DE full name DualEncoder, CE full name CrossEncoder):

<p align="center">
	<img src="./docs/dureader_retrieval_baselines.png" width="600" />
</p>

# 4. Model Training

Since the search problem will be divide into two stages of retrieval and reranking, the corresponding model optimization work is also focused in these two parts. For the retrieval model, we will use the classic asymmetric two-tower structure, combined with negative sample mining strategies to improve the  performance. As for reranking model, we will adopt a deep semantic ranking model based on a BERT-like pretrained model.

Before model training, you first need to create some datasets by the following command:

```bash
# make train data for point/pair/list-wise loss
sh script/make_train_data.sh

# make the training running time evaluation data
sh script/make_eval_data.sh
```

## 4.1 Retrieval Models

The traditional text retrieval system is implemented based on high-dimensional sparse text matching algorithms such as TFIDF/BM25. The main defect of such methods is that the text literal matching algorithm fails when the literal is inconsistent but the semantics are consistent. In recent years, with the rise of deep representation learning, methods based on low-dimensional dense vector matching have also been gradually introduced into retrieval systems. These methods generally use the dual-encoder two-tower model structure to perform vector matching of query and passage respectively. It means that the optimization objective during offline training expects example with the same semantics to be close in the vector space, while example with different semantics wants to be far apart in the vector space.

As for specific application scenarios, the dual-encoder two towers of the retrieval model may be **symmetric** or **asymmetric** structure, that depends on query and passage vectors if need to share a representation model. For the DuReaderRetrieval task, the asymmetric structure should be used, but we have made compatible considerations for the symmetric and asymmetric models in the code, which can smoothly switch to the symmetric task too.

The evaluation results of our models on **valid-set** as following:

| Model        | MRR@10 | recall@1 | recall@50 |
| ------------ | ------ | -------- | --------- |
| point-cosine | 14.02  | 8.2      | 52.05     |
| in-batch-neg | 25.58  | 16.90    | 74.40     |

Where `point-cosine` is the model trained with cosine loss based on offline negative sampling;  `In-batch-neg` is the model trained based on the online in-batch negative sampling strategy.

### 4.1.1 Offline Negatives Sampling

The idea of offline negative sampling is simple. Before the model training starts, global random sampling is performed to obtain negative samples. Here, the proportion of positive and negative samples needs to be controlled. The above dataset making process can be specified the negative ratio argument. The problem with this method is that the negative samples that may be obtained by global random sampling are easy negative, and the retrieval model lacks the ability to identify difficult negative samples.

After the offline global random sampling is done, the two-tower retrieval model can be optimized by point-wise or pair-wise methods. At present, we have implemented the following optimization methods:

- [CosineSimilarityLoss](./src/losses/cosine_similarity_loss.py)
- [ContrastiveLoss](./src/losses/contrastive_loss.py)
- [TripletLoss](./src/losses/triplet_loss.py)

Start train retrieval model as following:

```python
python src/train/train_dual_encoder.py --name train_de_loss_cosine \
    --model DMetaSoul/sbert-chinese-general-v2 \
    --dual-model DMetaSoul/sbert-chinese-general-v2 \
    --num-epochs 2 \
    --lr 3e-05 \
    --loss cosine \
    --train-file data/train/train.rand.neg.pair.tsv \
    --train-kind pair_with_label \
    --train-text-index 0,1 \
    --train-label-index 2 \
    --train-batch-size 64 \
    --save-steps 10000 \
    --eval-qid-file data/dev/dev4eval.qid.tsv \
    --eval-pid-file data/dev/dev4eval.pid.tsv \
    --eval-rel-file data/dev/dev4eval.rel.tsv \
```

Some important parameters are described here

- `--name` The id of training experiment
- `--model` The pre-trained model for query encoding
- `--dual-model` The pre-trained model for passage encoding
- `--loss` The optimization loss for training
- `--train-file` The training data file
- `--train-kind` The kind of training data, "pair", "pair_with_label", "triplet" and so on
- `--train-text-index` Which column of training data is text
- `--train-label-index` Which column of training data is label
- `--tied-model` If you want share the query and passage encoder, should specify it

For more training commands for other models, please refer to the script `script/train_dual_encoder.sh`.

When model training is done, you can run the following command to evaluate it on the valid-set:

```bash
# change to your model saved directory
q_model=output/train_de_loss_cosine/2022_05_27_21_16_29/epoch_1/model1
p_model=output/train_de_loss_cosine/2022_05_27_21_16_29/epoch_1/model2
query_file=./data/dev/dev.q.format
passage_data=./data/passage-collection
topk=50

# retrieval based passage database, will take some time
sh script/retrieval.sh ${q_model} ${p_model} ${query_file} ${passage_data} ${topk}

# evaluate the retrieval results
sh script/eval_retrieval.sh
```

### 4.1.2 Online Negatives Sampling

The idea of online negative sampling is to use other samples in the batch as negative samples during training (in-batch negatives sampling). Since each training step can have more negative samples, and those samples have been placed in the GPU, Therefore, the advantage of this method is that the training convergence is fast and the GPU memory utilization efficiency is high. At the same time, in-batch negatives sampling method can also be combined with the hard negative mining to further improve the performance.

Start train retrieval model as following:

```bash
python -u src/train/train_dual_encoder.py --name train_de_loss_contrastive_in_batch \
    --model DMetaSoul/sbert-chinese-general-v2 \
    --dual-model DMetaSoul/sbert-chinese-general-v2 \
    --num-epochs 2 \
    --lr 3e-05 \
    --loss contrastive_in_batch \
    --train-file data/train/train.pos.tsv \
    --train-kind pair \
    --train-text-index 0,1 \
    --train-label-index -1 \
    --train-batch-size 64 \
    --save-steps 2000 \
    --eval-qid-file data/dev/dev4eval.qid.tsv \
    --eval-pid-file data/dev/dev4eval.pid.tsv \
    --eval-rel-file data/dev/dev4eval.rel.tsv \
    > logs/train_dual_encoder-1.log 2>&1
```

For more training commands for other models, please refer to the script `script/train_dual_encoder.sh`. You can also use the above command (4.1.1) to evaluate this model.

## 4.2 Rerank Models

Due to the consideration of the performance of the retrieval model under large-scale data, the retrieval accuracy is often low. We will develop a deep semantic relevant model to re-rank the retrieval candidate results. Considering that the size of the reranking data is often small, an cross-based pre-training model with stronger learning ability can be used here, and combined with the training data for binary classification for rerank model optimization.

Since there are only positive samples available for training data, negative samples need to be constructed based on sampling. Below we will optimize according to different negative sample sampling methods. After the reranking model is trained, it is used to rank the results of the retrieval model, and the final results will be evaluated on the validation set.

At present, our reranking model performance on the **validation set**:

| Model                    | MRR@10 | recall@1 | recall@50 |
| ------------------------ | ------ | -------- | --------- |
| in-batch-neg (retrieval) | 25.58  | 16.90    | 74.40     |
| rand-neg5 (rerank)       | 24.45  | 14.80    | 74.40     |
| hard-neg1 (rerank)       | 47.28  | 34.45    | 74.40     |

It can be seen from the above that the performance of the global random negative sampling method (`rand-neg5`) is even worse than that of the retrieval model (`in-batch-neg`), mainly because the input data distribution of the reranking model is not globally uniform during inference, and It is consistent with the output data distribution of the recall model. `hard-neg1` is a method of randomly sampling negative samples for the candidate results of the retrievall model, and the performance is greatly improved (**25.58->47.28**).

Since the reranking model will sort the retrieval model results, it is necessary to generate the retrieval candidate results to be sorted first. The command is as follows:

```bash
# sh script/make_rerank_data.sh

python -u src/preprocess/make_rerank_data_from_recall_result.py \
    data/output/dev.recall.top50.json \
    data/dev/q2qid.dev.json \
    data/passage-collection/passage2id.map.json \
    data/passage-collection/part-00,data/passage-collection/part-01,data/passage-collection/part-02,data/passage-collection/part-03 \
    data/output/rerank.query-passage.pair.tsv
```

The retrieval candidate results will be saved in the `data/output/rerank.query-passage.pair.tsv` .

### 4.2.1 Global Random Negatives Sampling

The idea of global random negative sampling is simple. For each query, passage will be randomly sampled from the whole retrieval database as a negative sample, and the following commands will be run to generate training data: 

```bash
cat data/train/train.pos.tsv | python src/preprocess/negative_rand_sample.py 5 pair > data/train/train.rand.neg5.pair.tsv
```

Then start the model training with the following command:

```bash
# ref `script/train_cross_encoder.sh`

python -u src/train/train_cross_encoder.py --name train_ce_multiclass \
    --model DMetaSoul/sbert-chinese-general-v2 \
    --num-epochs 2 \
    --lr 3e-05 \
    --train-file data/train/train.rand.neg5.pair.tsv \
    --train-kind multiclass \
    --train-text-index 0,1 \
    --train-label-index 2 \
    --train-batch-size 32 \
    --num-labels 2 \
    --save-steps 2000 \
    --eval-qid-file data/dev/dev4eval.qid.tsv \
    --eval-pid-file data/dev/dev4eval.pid.tsv \
    --eval-rel-file data/dev/dev4eval.rel.tsv \
```

Some important arguments be noted as following

- `--train-kind` The type of task, can be `multiclass`, `multilabel`, `regression` and so on
- `--num-labels` The number of labels for mulitclass task

After the model training is done, you can evaluate the performance of the model on the validation set by running the following command:

```bash
# ref `script/run_rerank.sh`

# change to your model saved directory
model=output/train_ce_multiclass/2022_06_09_18_23_09/step_26000
pair_file=data/output/rerank.query-passage.pair.tsv
score_file=data/output/rerank.query-passage.pair.score

# rerank the retrieval candidates results 
sh script/rerank.sh ${model} ${pair_file} ${score_file}

# evaluate the rerank results
sh script/eval_rerank.sh
```

### 4.2.2 Negatives Sampling Based Retrieval Results

Because it is based on negative sampling of the retireval results, you first need to run the retireval results of all queries in the training data. The command is as follows:

```bash
# ref `script/run_retrieval.sh`

# change to your model saved directory
q_model=output/train_de_loss_cosine/2022_05_27_21_16_29/epoch_1/model1
p_model=output/train_de_loss_cosine/2022_05_27_21_16_29/epoch_1/model2
query_file=./data/train/train.q.format
passage_data=./data/passage-collection
topk=50

# retrieval based passage database
sh script/retrieval.sh ${q_model} ${p_model} ${query_file} ${passage_data} ${topk} train index
```

After the run is done, the retrieval candidates are saved in the `data/output/train.recall.top50` file.

Next, use the retrieval candidates to sample negative samples. The general idea is to randomly sample the non-positive samples in the retrieval results. The command is as follows:

```bash
# ref `script/make_hard_data.sh`

python src/preprocess/negative_hard_sample.py \
    data/train/train.pos.tsv \
    data/output/train.recall.top50 \
    data/passage-collection/part-00,data/passage-collection/part-01,data/passage-collection/part-02,data/passage-collection/part-03 \
    1 pair data/train/train.hard.neg1.pair.tsv \
```

The dataset generated by sampling will be saved in the `data/train/train.hard.neg1.pair.tsv` file, and then use this dataset for training:

```bash
python -u src/train/train_cross_encoder.py --name train_ce_multiclass \
    --model DMetaSoul/sbert-chinese-general-v2 \
    --num-epochs 2 \
    --lr 3e-05 \
    --train-file data/train/train.hard.neg1.pair.tsv \
    --train-kind multiclass \
    --train-text-index 0,1 \
    --train-label-index 2 \
    --train-batch-size 32 \
    --num-labels 2 \
    --save-steps 2000 \
    --eval-qid-file data/dev/dev4eval.qid.tsv \
    --eval-pid-file data/dev/dev4eval.pid.tsv \
    --eval-rel-file data/dev/dev4eval.rel.tsv \
```

The training command is the same as the above global negative sampling, the only difference is that the training data file is replaced here. When the training is completed, the above commands can also be used to evaluate the performance.

# 5. Model Inference

Both the retrieval model and the reranking model can perform online inference on [MetaSpore Serving](https://github.com/meta-soul/MetaSpore). Go to the model export tool [directory](https://github.com/meta-soul/MetaSpore/tree/main/demo/multimodal/offline/model_export), export and push the model to S3 cloud storage according to the following commands:

```bash
# export retrieval query model
rm -rf ./export

model_name=your-model-save-path/2022_05_27_18_53_58/epoch_1/model1
model_key=dense-retrieval-query-encoder

python src/modeling_export.py --exporter text_transformer_encoder --export-path ./export --model-name ${model_name} --model-key ${model_key} --raw-inputs texts --raw-preprocessor hf_tokenizer_preprocessor --raw-decoding json --raw-encoding arrow

s3_path=${MY_S3_PATH}/demo/nlp-algos-transformer/models/${model_key}
if [ $? == 0 ]; then
    aws s3 cp --recursive ./export ${s3_path}
fi

# export rerank model
rm -rf ./export

model_name=your-model-save-path/2022_06_08_19_11_59/epoch_0
model_key=deep-relevant-ranker

python src/modeling_export.py --exporter seq_transformer_classifier --export-path ./export --model-name ${model_name} --model-key ${model_key} --raw-inputs texts --raw-preprocessor hf_tokenizer_preprocessor --raw-decoding json --raw-encoding arrow

s3_path=${MY_S3_PATH}/demo/nlp-algos-transformer/models/${model_key}
if [ $? == 0 ]; then
    aws s3 cp --recursive ./export ${s3_path}
fi
```

