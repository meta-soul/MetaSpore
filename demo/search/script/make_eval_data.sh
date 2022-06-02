# raw json to tsv
cat data/dev/dev.json | python src/preprocess/json2tsv.py > data/dev/dev.tsv

# make relevant data for training evaluation
python src/preprocess/pair2relevant.py data/dev/dev.tsv \
    data/dev/dev4eval.qid.tsv \
    data/dev/dev4eval.pid.tsv \
    data/dev/dev4eval.rel.tsv \
#cut -f1 data/dev/dev4eval.qid.tsv > data/dev/dev.qid
