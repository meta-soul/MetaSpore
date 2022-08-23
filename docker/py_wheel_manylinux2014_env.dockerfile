FROM quay.io/pypa/manylinux2014_x86_64

RUN yum update -y && yum install -y devtoolset-11-gcc-c++ curl zip unzip tar perl-IPC-Cmd flex && yum clean packages
ENV DEVTOOLSET_ROOTPATH=/opt/rh/devtoolset-11/root
ENV PATH /opt/rh/devtoolset-11/root/usr/bin:${PATH}
ENV MANPATH /opt/rh/devtoolset-11/root/usr/share/man:${MANPATH}
ENV INFOPATH /opt/rh/devtoolset-11/root/usr/share/info:${INFOPATH}
ENV PCP_DIR /opt/rh/devtoolset-11/root
ENV LD_LIBRARY_PATH /opt/rh/devtoolset-11/root/usr/lib64:/opt/rh/devtoolset-11/root/usr/lib:/opt/rh/devtoolset-11/root/usr/lib64/dyninst:/opt/rh/devtoolset-11/root/usr/lib/dyninst:${LD_LIBRARY_PATH}
ENV PKG_CONFIG_PATH /opt/rh/devtoolset-11/root/usr/lib64/pkgconfig:${PKG_CONFIG_PATH}
RUN git clone https://github.com/Microsoft/vcpkg.git /opt/vcpkg
RUN /opt/vcpkg/bootstrap-vcpkg.sh
COPY vcpkg-wheel.json /opt/vcpkg.json
RUN echo "set(VCPKG_C_FLAGS \"-D_GLIBCXX_USE_CXX11_ABI=0\")" >> /opt/vcpkg/triplets/x64-linux.cmake
RUN echo "set(VCPKG_CXX_FLAGS \"-D_GLIBCXX_USE_CXX11_ABI=0\")" >> /opt/vcpkg/triplets/x64-linux.cmake
RUN echo "set(VCPKG_BUILD_TYPE release)" >> /opt/vcpkg/triplets/x64-linux.cmake
RUN PATH=/opt/python/cp38-cp38/bin:$PATH && LD_LIBRARY_PATH=/opt/python/cp38-cp38/lib:$LD_LIBRARY_PATH && /opt/vcpkg/vcpkg install --x-install-root=/opt/vcpkg_installed --x-manifest-root=/opt --clean-after-build
