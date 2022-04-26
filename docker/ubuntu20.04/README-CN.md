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
    DOCKER_BUILDKIT=1 docker build --network=host -f docker/ubuntu20.04/Dockerfile_dev -t $REPOSITORY/metaspore-dev:v1.0.0 .
    ```

    1. Serving Build 镜像，基于 Dev 镜像生成 Serving 服务构建结果：`Dockerfile_serving_build`
        ```bash
        DOCKER_BUILDKIT=1 docker build --network=host -f docker/ubuntu20.04/Dockerfile_serving_build --build-arg DEV_IMAGE=$REPOSITORY/metaspore-dev:v1.0.0 -t $REPOSITORY/metaspore-serving-build:v1.0.0 .
        ```
        1. Serving Release 镜像：基于 Serving Build 镜像，生成可发布的镜像，strip 了 Debug 段以减小镜像体积：`Dockerfile_serving_release`
            ```bash
            DOCKER_BUILDKIT=1 docker build --network=host -f docker/ubuntu20.04/Dockerfile_serving_release --build-arg BUILD_IMAGE=$REPOSITORY/metaspore-serving-build:v1.0.0 -t $REPOSITORY/metaspore-serving-release:v1.0.0 --target serving_release .
            ```
        1. Serving Debug 镜像：基于 Serving Build 镜像，生成 Debug Info 和携带 GDB 环境的镜像：`Dockerfile_serving_release`
            ```bash
            DOCKER_BUILDKIT=1 docker build --network=host -f docker/ubuntu20.04/Dockerfile_serving_release --build-arg BUILD_IMAGE=$REPOSITORY/metaspore-serving-build:v1.0.0 -t $REPOSITORY/metaspore-serving-debug:v1.0.0 --target serving_debug .
            ```

    1. Training Build 镜像，基于 Dev 镜像生成 Training Wheel 安装包：`Dockerfile_training_build`

        1. Training Release 镜像：基于 Training Build，生成训练镜像，包含 Spark 等依赖：`Dockerfile_training_release`
            ```bash

            ```