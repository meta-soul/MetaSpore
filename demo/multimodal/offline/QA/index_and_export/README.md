# Usage

## Data Indexing

This section explains how to create an indexing for online retrieval based on the raw material database, which is roughly divided into the following steps:

1. **preprocess**

    ```
    sh preprocess.sh
    ```

2. **indexing**

    ```
    sh build.sh
    ```

3. **pushing**

    ```
    sh push_milvus.sh
    
    sh push_mongo.sh
```

## Model Export

We need export the above indexing model for online services:

```
sh export.sh
```

The onnx model and tokenizer will be exported and saved into S3 bucket. And then the online services can load them.
