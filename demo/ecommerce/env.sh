if [ ! -f $(dirname ${BASH_SOURCE[0]})/python-env/requirements.txt ]
then
    $(dirname ${BASH_SOURCE[0]})/env.restore.sh
fi

source $(realpath $(dirname ${BASH_SOURCE[0]}))/python-env/bin/activate

export MONGODB_IMAGE="mongo:5.0"
export MYSQL_IMAGE="mysql:5.7"
export OFFLINE_IMAGE="swr.cn-southwest-2.myhuaweicloud.com/dmetasoul-repo/metaspore-training-release:v1.1.1"
export ONLINE_IMAGE="swr.cn-southwest-2.myhuaweicloud.com/dmetasoul-repo/recommend-service:1.0.0"
export FRONT_IMAGE="swr.cn-southwest-2.myhuaweicloud.com/dmetasoul-repo/ecommerce-vue-app:v1.0.0"
export SERVING_IMAGE="swr.cn-southwest-2.myhuaweicloud.com/dmetasoul-repo/metaspore-serving-release:cpu-v1.0.1"
export CONSUL_IMAGE="consul:1.13.1"
