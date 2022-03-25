#
# To run locally, execute:
#
#   spark-submit --master local[2] wide_and_deep_example.py
#

S3_ROOT_DIR = 's3://{YOUR_S3_BUCKET}/{YOUR_S3_PATH}/'

batch_size = 100
worker_count = 1
server_count = 1

import metaspore as ms
spark = ms.spark.get_session(batch_size=batch_size,
                             worker_count=worker_count,
                             server_count=server_count,
                            )
sc = spark.sparkContext

with spark:
    module = ms.nn.WideAndDeepModule(
        wide_column_name_path=S3_ROOT_DIR + 'demo/schema/column_name_demo.txt',
        wide_combine_schema_path=S3_ROOT_DIR + 'demo/schema/combine_schema_demo.txt',
        deep_sparse_column_name_path=S3_ROOT_DIR + 'demo/schema/column_name_demo.txt',
        deep_sparse_combine_schema_path=S3_ROOT_DIR + 'demo/schema/combine_schema_demo.txt',
    )

    model_out_path = S3_ROOT_DIR + 'demo/output/dev/model_out/'
    estimator = ms.PyTorchEstimator(module=module,
                                    worker_count=worker_count,
                                    server_count=server_count,
                                    model_out_path=model_out_path,
                                    input_label_column_index=0)

    train_dataset_path = S3_ROOT_DIR + 'demo/data/train/day_0_0.001_train.csv'
    train_dataset = ms.input.read_s3_csv(spark, train_dataset_path, delimiter='\t')
    model = estimator.fit(train_dataset)

    test_dataset_path = S3_ROOT_DIR + 'demo/data/test/day_0_0.001_test.csv'
    test_dataset = ms.input.read_s3_csv(spark, test_dataset_path, delimiter='\t')
    result = model.transform(test_dataset)
    result.show(5)

    import pyspark
    evaluator = pyspark.ml.evaluation.BinaryClassificationEvaluator()
    test_auc = evaluator.evaluate(result)
    print('test_auc: %g' % test_auc)
