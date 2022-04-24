model_key=$1
if [ -z "$model_key" ]; then
    echo "model key is empty!"
    exit
fi
dev_file=$2
if [ ! -f "$dev_file" ]; then
    dev_file=../../datasets/processed/Chinese-STS-B/dev.tsv
fi

#dev_file=../../datasets/processed/afqmc_public/dev.tsv
#test_file=../../datasets/processed/afqmc_public/dev.tsv
#dev_file=../../datasets/processed/lcqmc/dev.tsv
#test_file=../../datasets/processed/lcqmc/dev.tsv

# BERT-mean
python bert_base.py --name BERT-mean --eval-file $dev_file
#python bert_base.py --name BERT-mean_C-STS-B-test --eval-file $test_file

# BERT-whitening
python bert_whitening.py --name BERT-white --eval-file $dev_file
#python bert_whitening.py --name BERT-white_C-STS-B-test --eval-file $test_file

# SimBERT-mean
python bert_base.py --name SimBERT-mean --model WangZeJun/simbert-base-chinese --eval-file $dev_file
#python bert_base.py --name SimBERT-mean_C-STS-B-test --model WangZeJun/simbert-base-chinese --eval-file $test_file

# FastText
data_file=$dev_file
t1_file=${model_key}.fasttext.t1.txt
t2_file=${model_key}.fasttext.t2.txt
seg1_file=${model_key}.fasttext.seg1.txt
seg2_file=${model_key}.fasttext.seg2.txt
e1_file=${model_key}.fasttext.e1.txt
e2_file=${model_key}.fasttext.e2.txt
s1_file=${model_key}.fasttext.s1.txt
s2_file=${model_key}.fasttext.s2.txt
ft_dir=~/tools/fastText-0.9.2
seg_script=~/tools/stanford-segmenter-2020-11-17/segment.sh
cut -f1 $data_file > $t1_file
cut -f2 $data_file > $t2_file
cut -f3 $data_file > $s1_file
sh ${seg_script} ctb $t1_file UTF-8 0 > $seg1_file
sh ${seg_script} ctb $t2_file UTF-8 0 > $seg2_file
cat $seg1_file | ${ft_dir}/fasttext print-sentence-vectors ${ft_dir}/models/cc.zh.300.bin > $e1_file
cat $seg2_file | ${ft_dir}/fasttext print-sentence-vectors ${ft_dir}/models/cc.zh.300.bin > $e2_file
python tools/emb_similarity.py $e1_file $e2_file > $s2_file
echo "fastText_Chinese-STS-B Dev spearman: "
python tools/corr.py $s1_file $s2_file spearman
rm -f $t1_file $seg1_file $e1_file $s1_file $t2_file $seg2_file $e2_file $s2_file

:<<EOF
data_file=$test_file
cut -f1 $data_file > $t1_file
cut -f2 $data_file > $t2_file
cut -f3 $data_file > $s1_file
sh ${seg_script} ctb $t1_file UTF-8 0 > $seg1_file
sh ${seg_script} ctb $t2_file UTF-8 0 > $seg2_file
cat $seg1_file | ${ft_dir}/fasttext print-sentence-vectors ${ft_dir}/models/cc.zh.300.bin > $e1_file
cat $seg2_file | ${ft_dir}/fasttext print-sentence-vectors ${ft_dir}/models/cc.zh.300.bin > $e2_file
python ../tools/emb_similarity.py $e1_file $e2_file > $s2_file
echo "fastText_Chinese-STS-B Test spearman: "
python ../tools/corr.py $s1_file $s2_file spearman
rm -f $t1_file $seg1_file $e1_file $s1_file $t2_file $seg2_file $e2_file $s2_file
EOF

# Tencent embedding
label_file=${model_key}.dev.labels
score_file=${model_key}.dev.scores
cut -f3 $dev_file > ${label_file}
sh tencent_emb.sh $dev_file > ${score_file}
echo "Tencent_Chinese-STS-B Dev spearman: "
python tools/corr.py ${label_file} ${score_file} spearman
rm -rf ${label_file} ${score_file}
:<<EOF
echo "Tencent_Chinese-STS-B Test spearman: "
cut -f3 $test_file > test.labels
sh tencent_emb.sh $test_file > test.scores
python ../tools/corr.py test.labels test.scores spearman
rm -f test.labels test.scores
EOF
