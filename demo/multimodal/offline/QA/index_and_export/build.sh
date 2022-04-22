export PYTHONPATH="${PYTHONPATH}:./src"

#python src/indexing/text_build.py --model DMetaSoul/sbert-chinese-qmc-domain-v1 --doc-file data/baike/baike_qa_1w.doc.json --index-file data/baike/baike_qa_1w.doc.index.json --batch-size 16

nohup python src/indexing/text_build.py --model DMetaSoul/sbert-chinese-qmc-domain-v1 --doc-file data/baike/baike_qa_train.doc.json --index-file data/baike/baike_qa_train.doc.index.json --batch-size 128 --device cuda:0 > build.log 2>&1 &
