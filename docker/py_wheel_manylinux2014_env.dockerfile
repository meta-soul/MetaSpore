FROM quay.io/pypa/manylinux2014_x86_64

RUN yum update -y && yum install -y curl zip unzip tar perl-IPC-Cmd flex && yum clean packages
RUN git clone https://github.com/Microsoft/vcpkg.git /opt/vcpkg
RUN /opt/vcpkg/bootstrap-vcpkg.sh
COPY vcpkg-wheel.json /opt/vcpkg.json
RUN echo "set(VCPKG_BUILD_TYPE release)" >> /opt/vcpkg/triplets/x64-linux.cmake
RUN PATH=/opt/python/cp38-cp38/bin:$PATH && LD_LIBRARY_PATH=/opt/python/cp38-cp38/lib:$LD_LIBRARY_PATH && /opt/vcpkg/vcpkg install --x-install-root=/opt/vcpkg_installed --x-manifest-root=/opt --clean-after-build