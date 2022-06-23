python tools/moviedata_scheme_data.py \
  --rating-data data/moviedata-10m-split/train.csv \
  --movies-data data/moviedata-10m-split/movie.json \
  --user-data data/moviedata-10m-schema/user.train.json \
  --item-data data/moviedata-10m-schema/item.train.json \
  --action-data data/moviedata-10m-schema/action.train.json

python tools/moviedata_scheme_data.py \
  --rating-data data/moviedata-10m-split/test.csv \
  --movies-data data/moviedata-10m-split/movie.json \
  --user-data data/moviedata-10m-schema/user.test.json \
  --item-data data/moviedata-10m-schema/item.test.json \
  --action-data data/moviedata-10m-schema/action.test.json
