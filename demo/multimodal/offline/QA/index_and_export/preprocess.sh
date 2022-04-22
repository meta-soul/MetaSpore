#python src/preprocess/make_baike_qa_data.py --input-jsonline data/baike/baike_qa_1w.json --output-jsonline data/baike/baike_qa_1w.doc.json --start-id 0
nohup python src/preprocess/make_baike_qa_data.py --input-jsonline data/baike/baike_qa_train.json --output-jsonline data/baike/baike_qa_train.doc.json --start-id 0 > preprocess.log 2>&1 &
