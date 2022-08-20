# Docker 镜像构建文档

提供了基于 Ubuntu:20.04 的 Docker 镜像构建，包括离线训练镜像、在线 Serving 服务镜像，镜像能够直接部署到 K8s 环境运行。

## 镜像构建层次
以下 docker build 命令均在 project 根目录执行.

首先设置镜像仓库地址：

```bash
export REPOSITORY=...
```

1. Dev 镜像，打包了基础环境、在离线 C++、Python 编译构建环境：`Dockerfile_dev`
    ```bash
    DOCKER_BUILDKIT=1 docker build --network=host --build-arg HTTP_PROXY=${http_proxy} --build-arg HTTPS_PROXY=${https_proxy} -f docker/ubuntu20.04/Dockerfile_dev -t $REPOSITORY/metaspore-dev:v1.0.0 .
    ```
    可以通过 `--build-arg RUNTIME=gpu` 构建 GPU 支持的 dev 镜像。默认为 CPU 环境。

    1. Serving Build 镜像，基于 Dev 镜像生成 Serving 服务构建结果：`Dockerfile_serving_build`
        ```bash
        DOCKER_BUILDKIT=1 docker build --network=host -f docker/ubuntu20.04/Dockerfile_serving_build --build-arg DEV_IMAGE=$REPOSITORY/metaspore-dev:v1.0.0 -t $REPOSITORY/metaspore-serving-build:v1.0.0 .
        ```
        可以通过 `--build-arg ENABLE_GPU=ON` 构建支持 GPU 的 serving 服务。默认只支持 CPU 预测。

        1. Serving Release 镜像：基于 Serving Build 镜像，生成可发布的镜像，strip 了 Debug 段以减小镜像体积：`Dockerfile_serving_release`
            ```bash
            DOCKER_BUILDKIT=1 docker build --network=host -f docker/ubuntu20.04/Dockerfile_serving_release --build-arg BUILD_IMAGE=$REPOSITORY/metaspore-serving-build:v1.0.0 -t $REPOSITORY/metaspore-serving-release:v1.0.0 --target serving_release .
            ```
            可以通过 `--build-arg RUNTIME=gpu` 构建 GPU 支持的 release 镜像。默认只支持 CPU。
        1. Serving Debug 镜像：基于 Serving Build 镜像，生成 Debug Info 和携带 GDB 环境的镜像：`Dockerfile_serving_release`
            ```bash
            DOCKER_BUILDKIT=1 docker build --network=host -f docker/ubuntu20.04/Dockerfile_serving_release --build-arg BUILD_IMAGE=$REPOSITORY/metaspore-serving-build:v1.0.0 -t $REPOSITORY/metaspore-serving-debug:v1.0.0 --target serving_debug .
            ```

    1. Training Build 镜像，基于 Dev 镜像生成 Training Wheel 安装包：`Dockerfile_training_build`
        ```bash
        DOCKER_BUILDKIT=1 docker build --network=host -f docker/ubuntu20.04/Dockerfile_training_build --build-arg DEV_IMAGE=$REPOSITORY/metaspore-dev:v1.0.0 -t $REPOSITORY/metaspore-training-build:v1.0.0 .
        ```
        1. Training Release 镜像：基于 Training Build，生成训练镜像，包含 Spark 等依赖：`Dockerfile_training_release`。该镜像构建支持多种选项以安装 MetaSpore 和 Spark，默认从 http 下载它们的安装包：
            ```bash
            DOCKER_BUILDKIT=1 docker build --network=host -f docker/ubuntu20.04/Dockerfile_training_release -t $REPOSITORY/metaspore-training-release:v1.0.0 --target release .

            # 可以通过 --build-arg METASPORE_WHEEL="http://..." 和 --build-arg SPARK_FILE="http://" 来指定 MetaSpore 的 wheel 包路径和 Spark 的 tgz 安装包路径
            ```

            1. 从 MetaSpore build 镜像生成 Release 镜像：
                ```bash
                # 通过 METASPORE_RELEASE=build 指定从 build 镜像安装 MetaSpore，同时需要通过 METASPORE_BUILD_IMAGE 指定镜像名。
                DOCKER_BUILDKIT=1 docker build --network=host -f docker/ubuntu20.04/Dockerfile_training_release --build-arg METASPORE_RELEASE=build --build-arg METASPORE_BUILD_IMAGE=$REPOSITORY/metaspore-training-build:v1.0.0 -t $REPOSITORY/metaspore-training-release:v1.0.0 --target release .
                ```
            2. 使用 PySpark 安装 Spark 环境：
                ```bash
                DOCKER_BUILDKIT=1 docker build --network=host -f docker/ubuntu20.04/Dockerfile_training_release --build-arg METASPORE_RELEASE=build --build-arg METASPORE_BUILD_IMAGE=$REPOSITORY/metaspore-training-build:v1.0.0 --build-arg SPARK_RELEASE=pyspark --build-arg SPARK_FILE="pyspark==3.2.1" -t $REPOSITORY/metaspore-training-release:v1.0.0 --target release .
                ```
            3. 构建 Jupyter 镜像：
                Jupyter 镜像需要使用 PySpark 方式安装的 Training Release 镜像：
                ```bash
                DOCKER_BUILDKIT=1 docker build --network=host -f docker/ubuntu20.04/Dockerfile_jupyter --build-arg RELEASE_IMAGE=$REPOSITORY/metaspore-training-release:v1.0.0 -t $REPOSITORY/metaspore-training-jupyter:v1.0.0 docker/ubuntu20.04
                ```
                在 Jupyter 镜像构建中，已经内置了代码自动补全、可视化、Pipeline 等常用插件。
