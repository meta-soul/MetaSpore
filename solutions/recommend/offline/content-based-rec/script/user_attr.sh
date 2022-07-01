python -m jobs.user_model.user_offline_tag \
    --job-name "user-tag" \
    --scene-id ml-cb-100 \
    --action-type rating \
    --action-data ../data/ml-1m-schema/action.train \
    --item-data ../data/ml-1m-dump/item.attr \
    --dump-data ../data/ml-1m-dump/user.tag \
    --action-max-len 30 \
    --action-sortby-key action_time

python -m jobs.tools.push_mongo --mongo-uri mongodb://172.31.37.47:27017 \
    --mongo-database jpa \
    --mongo-collection movielens_cb_demo_user_tag \
    --data ../data/ml-1m-dump/user.tag \
    --fields user_id,tags,tag_weights \
    --index-fields user_id \
    --write-mode overwrite

python -m jobs.tools.push_mongo --mongo-uri mongodb://172.31.37.47:27017 \
    --mongo-database jpa \
    --mongo-collection movielens_cb_demo_user_emb \
    --data ../data/ml-1m-dump/user.emb \
    --fields user_id,user_emb \
    --index-fields user_id \
    --write-mode overwrite
