python tools/movielens_scheme_data.py \
  --rating-data data/ml-1m-split/train.csv \
  --movies-data data/ml-1m/movies.dat \
  --user-data data/ml-1m-schema/user.train.json \
  --item-data data/ml-1m-schema/item.train.json \
  --action-data data/ml-1m-schema/action.train.json

python tools/movielens_scheme_data.py \
  --rating-data data/ml-1m-split/test.csv \
  --movies-data data/ml-1m/movies.dat \
  --user-data data/ml-1m-schema/user.test.json \
  --item-data data/ml-1m-schema/item.test.json \
  --action-data data/ml-1m-schema/action.test.json
