ARG DEV_IMAGE
FROM ${DEV_IMAGE} as build_serving
ARG ENABLE_GPU=OFF

RUN python -m pip install grpcio-tools --no-cache-dir

COPY . /opt/metaspore-src

RUN --mount=type=cache,target=/opt/vcpkg_installed cmake -B /opt/metaspore-build-release -S /opt/metaspore-src -DCMAKE_INSTALL_PREFIX=/opt/metaspore-serving \
    -DCMAKE_PREFIX_PATH="/opt/vcpkg_installed/x64-linux;/usr/local/cuda" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DENABLE_TESTS=OFF -DBUILD_SERVING_BIN=ON -DBUILD_TRAIN_PKG=OFF -DENABLE_GPU=${ENABLE_GPU}
RUN --mount=type=cache,target=/opt/vcpkg_installed cmake --build /opt/metaspore-build-release --target install -- -j8
RUN --mount=type=cache,target=/opt/vcpkg_installed mkdir -p /opt/metaspore/debug
RUN --mount=type=cache,target=/opt/vcpkg_installed objcopy --only-keep-debug /opt/metaspore-serving/bin/metaspore-serving-bin /opt/metaspore/debug/metaspore-serving-bin.debug
RUN --mount=type=cache,target=/opt/vcpkg_installed objcopy --strip-debug --add-gnu-debuglink=/opt/metaspore/debug/metaspore-serving-bin.debug /opt/metaspore-serving/bin/metaspore-serving-bin /opt/metaspore-serving/bin/metaspore-serving-bin