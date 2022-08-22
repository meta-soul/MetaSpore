# Docker image build documentation

Provides Docker image construction based on Ubuntu:20.04, including offline training image, online Serving service image, and the image can be directly deployed to the K8s environment to run.

## Image build hierarchy
The following docker build commands are executed in the project root directory.

First set the mirror warehouse address:

```bash
export REPOSITORY=...
````

1. Dev image, package the basic environment, build the environment for C++ and Python compilation: `Dockerfile_dev`
    ```bash
    DOCKER_BUILDKIT=1 docker build --network host --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy} -f docker/ubuntu20.04/Dockerfile_dev -t $REPOSITORY/metaspore-dev:v1.0.0 .
    ````
    Use `--build-arg RUNTIME=gpu` to enable GPU dev image. Default is CPU only.

    1. Serving Build image, generate Serving service build result based on Dev image: `Dockerfile_serving_build`
        ```bash
        DOCKER_BUILDKIT=1 docker build --network host --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy} -f docker/ubuntu20.04/Dockerfile_serving_build --build-arg DEV_IMAGE=$REPOSITORY/metaspore-dev:v1.0.0 -t $REPOSITORY/metaspore-serving-build:v1.0.0 .
        ````
        Use `--build-arg ENABLE_GPU=ON` to build Serving service with GPU support. Default is CPU only.
        1. Serving Release image: Generate a releasable image based on the Serving Build image, strip the Debug section to reduce the image size: `Dockerfile_serving_release`
            ```bash
            DOCKER_BUILDKIT=1 docker build --network host --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy} -f docker/ubuntu20.04/Dockerfile_serving_release --build-arg BUILD_IMAGE=$REPOSITORY/metaspore-serving-build:v1.0.0 -t $REPOSITORY/metaspore-serving-release:v1 .0.0 --target serving_release .
            ````
            Use `--build-arg RUNTIME=gpu` to build Serving release image with GPU support. Default is CPU only.
        1. Serving Debug image: Based on the Serving Build image, generate Debug Info and an image carrying the GDB environment: `Dockerfile_serving_release`
            ```bash
            DOCKER_BUILDKIT=1 docker build --network host --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy} -f docker/ubuntu20.04/Dockerfile_serving_release --build-arg BUILD_IMAGE=$REPOSITORY/metaspore-serving-build:v1.0.0 -t $REPOSITORY/metaspore-serving-debug:v1 .0.0 --target serving_debug .
            ````

    1. Training Build image, generate Training Wheel installation package based on Dev image: `Dockerfile_training_build`
        ```bash
        DOCKER_BUILDKIT=1 docker build --network host --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy} -f docker/ubuntu20.04/Dockerfile_training_build --build-arg DEV_IMAGE=$REPOSITORY/metaspore-dev:v1.0.0 -t $REPOSITORY/metaspore-training-build:v1.0.0 .
        ````
        1. Training Release image: Based on Training Build, generate a training image, including Spark and other dependencies: `Dockerfile_training_release`. The image build supports several options to install MetaSpore and Spark, and download their packages from http by default:
            ```bash
            DOCKER_BUILDKIT=1 docker build --network host --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy} -f docker/ubuntu20.04/Dockerfile_training_release -t $REPOSITORY/metaspore-training-release:v1.0.0 --target release .

            # You can specify MetaSpore's wheel package path and Spark's tgz installation package path through --build-arg METASPORE_WHEEL="http://..." and --build-arg SPARK_FILE="http://"
            ````

            1. Generate the Release image from the MetaSpore build image:
                ```bash
                # Specify to install MetaSpore from the build image through METASPORE_RELEASE=build, and you need to specify the image name through METASPORE_BUILD_IMAGE.
                DOCKER_BUILDKIT=1 docker build --network host --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy} -f docker/ubuntu20.04/Dockerfile_training_release --build-arg METASPORE_RELEASE=build --build-arg METASPORE_BUILD_IMAGE=$REPOSITORY/metaspore-training-build:v1.0.0 -t $REPOSITORY /metaspore-training-release:v1.0.0 --target release .
                ````
            2. Install the Spark environment using PySpark:
                ```bash
                DOCKER_BUILDKIT=1 docker build --network host --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy} -f docker/ubuntu20.04/Dockerfile_training_release --build-arg METASPORE_RELEASE=build --build-arg METASPORE_BUILD_IMAGE=$REPOSITORY/metaspore-training-build:v1.0.0 --build- arg SPARK_RELEASE=pyspark --build-arg SPARK_FILE="pyspark==3.2.1" -t $REPOSITORY/metaspore-training-release:v1.0.0 --target release .
                ````
            3. Generate Jupyter Image：
                Jupyter image requires Training Release build with pyspark installation.
                ```bash
                DOCKER_BUILDKIT=1 docker build --network host --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy} -f docker/ubuntu20.04/Dockerfile_jupyter --build-arg RELEASE_IMAGE=$REPOSITORY/metaspore-training-release:v1.0.0 -t $REPOSITORY/metaspore-training-jupyter:v1.0.0 docker/ubuntu20.04
                ```
