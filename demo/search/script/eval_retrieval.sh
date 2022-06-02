QUERY2ID=./data/dev/q2qid.dev.json
PARA2ID=./data/passage-collection/passage2id.map.json
MODEL_OUTPUT=./data/output/dev.recall.top50
REFERENCE_FIEL=./data/dev/dev.json
PREDICTION_FILE=./data/output/dev.recall.top50.json

python src/eval/convert_recall_res_to_json.py ${QUERY2ID} ${PARA2ID} ${MODEL_OUTPUT} ${PREDICTION_FILE}

python src/eval/evaluation.py $REFERENCE_FIEL $PREDICTION_FILE
