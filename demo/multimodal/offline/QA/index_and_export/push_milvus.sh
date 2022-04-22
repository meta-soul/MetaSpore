export PYTHONPATH="${PYTHONPATH}:./src"
source ./env.sh

host=${MY_MILVUS_HOST}
port=${MY_MILVUS_PORT}
collection=baike_qa_demo
# 删除索引库
#python src/indexing/milvus/drop.py --host ${host} --port ${port} --collection-name ${collection}

# 构建索引库
#python src/indexing/milvus/push.py --host ${host} --port ${port} --collection-name ${collection} --index-field question_emb  --index-file data/baike/baike_qa_1w.doc.index.json
nohup python src/indexing/milvus/push.py --host ${host} --port ${port} --collection-name ${collection} --index-field question_emb  --index-file data/baike/baike_qa_train.doc.index.json > push_milvus.log 2>&1 &
