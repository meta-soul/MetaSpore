python -u train.py --name title_to_fashion \
    --model bert-base-cased --num-labels 1 \
    --train-file /your/working/path/title_to_fashion_500k/train.tsv \
    --eval-file /your/working/path/title_to_fashion_500k/val.tsv \
    --eval-steps 1000 \
    --num-epochs 1 --lr 2e-5 --train-batch-size 32 --eval-batch-size 32 --gpu 0 \
    --output-path ./output
