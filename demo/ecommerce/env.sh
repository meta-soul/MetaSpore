if [ ! -f $(dirname ${BASH_SOURCE[0]})/python-env/requirements.txt ]
then
    $(dirname ${BASH_SOURCE[0]})/env.restore.sh
fi

source $(realpath $(dirname ${BASH_SOURCE[0]}))/python-env/bin/activate

export MONGODB_IMAGE="mongo:5.0@sha256:dc8252645a3f6bad272f54ae488767d270f7c26a8bba6d76d3be69e423fe2eba"
export MYSQL_IMAGE="mysql:5.7@sha256:3e704854fb64969e551bf2a17d4e804778d26848e3b61533a415c7dc5711f2e7"
export OFFLINE_IMAGE="swr.cn-southwest-2.myhuaweicloud.com/dmetasoul-repo/metaspore-training-release:v1.1.1@sha256:5bd836fbbb08ab1428cf4c5dfded9a40d5d64d1bd281b567e1e89c01f4924fcf"
export ONLINE_IMAGE="swr.cn-southwest-2.myhuaweicloud.com/dmetasoul-repo/recommend-service:1.0.0@sha256:f62363f9dfc95ce7a4fcad59245292dc13135144fa5145de57e6c6ec5457913c"
export FRONT_IMAGE="swr.cn-southwest-2.myhuaweicloud.com/dmetasoul-repo/ecommerce-vue-app:v1.0.0@sha256:e3dc2cda9840980d34187c49c41308bd441ad90f270be2a1b90a87295876b4bc"
export SERVING_IMAGE="swr.cn-southwest-2.myhuaweicloud.com/dmetasoul-repo/metaspore-serving-release:cpu-v1.0.1@sha256:99b62896bf2904b1e2814eb247e1e644f83b9c90128454d96261088bb24ec80a"
export CONSUL_IMAGE="consul:1.13.1@sha256:4f54d5ddb23771cf79d9ad543d1e258b7da802198bc5dbc3ff85992cc091a50e"
