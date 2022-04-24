text_file=$1
if [ ! -f "${text_file}" ]; then
    echo "text file not exists!"
    exit
fi

model_file=~/tools/TencentEmbedding/tencent-ailab-embedding-zh-d200-v0.2.0-s/tencent-ailab-embedding-zh-d200-v0.2.0-s.txt
python tencent_emb.py $text_file $model_file
