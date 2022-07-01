python -m jobs.item_model.item_offline_embed \
    --job-name item-embedding \
    --item-data ../data/ml-1m-schema/item.train \
    --dump-data ../data/ml-1m-dump/item.emb \
    --text-model-name sentence-transformers/all-MiniLM-L6-v2 \
    --batch-size 64 \
    --device cuda:0

python -m jobs.tools.push_mongo --mongo-uri mongodb://172.31.37.47:27017 \
    --mongo-database jpa \
    --mongo-collection movielens_cb_demo_item_embid \
    --data ../data/ml-1m-dump/item.emb \
    --fields id,item_id \
    --index-fields id \
    --write-mode overwrite

python -m jobs.tools.push_milvus --milvus-host 120.92.77.120 \
    --milvus-port 19530 \
    --milvus-collection movielens_cb_demo_item_emb \
    --data ../data/ml-1m-dump/item.emb \
    --fields id,item_emb \
    --id-field id \
    --emb-field item_emb \
    --write-batch 1024 \
    --write-interval 0.1 \
    --index-type IVF_FLAT \
    --index-metric IP \
    --index-nlist 1024
