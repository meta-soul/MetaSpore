# raw json to tsv data
cat data/train/train.json | python src/preprocess/json2tsv.py > data/train/train.tsv

# [(query, pos),...]
cut -f1,3 data/train/train.tsv > data/train/train.pos.tsv

# [(query, pos, 1), (query, neg, 0),...]
cat data/train/train.pos.tsv | python src/preprocess/negative_rand_sample.py 5 pair > data/train/train.rand.neg.pair.tsv

# [(query, pos, neg),...]
cat data/train/train.pos.tsv | python src/preprocess/negative_rand_sample.py 1 triplet > data/train/train.rand.neg.triplet.tsv
