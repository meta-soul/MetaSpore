# Build offline training python wheel package

## Environment Setup
Ubuntu 20.04 is recommended as the development environment. The MetaSpore training framework is developed in C++ 20 and provides a Python interface.

First install the development dependencies: 
```bash
sudo apt install build-essential manpages-dev software-properties-common curl zip unzip tar pkg-config bison flex python3-dev
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt install gcc-11 g++-11

# optional steps if you have multiple versions of gcc/g++
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 --slave /usr/bin/g++ g++ /usr/bin/g++-9 --slave /usr/bin/gcov gcov /usr/bin/gcov-9 --slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-9 --slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-9
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110 --slave /usr/bin/g++ g++ /usr/bin/g++-11 --slave /usr/bin/gcov gcov /usr/bin/gcov-11 --slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-11 --slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-11

# install latest cmake
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null 
sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" 
sudo apt update
sudo apt install cmake

# MetaSpore uses vcpkg to manage thirdparty c++ dependencies
git clone https://github.com/Microsoft/vcpkg.git ~/.vcpkg
~/.vcpkg/bootstrap-vcpkg.sh
```

## Compile and build
```bash
git clone https://github.com/meta-soul/MetaSpore.git
cd MetaSpore
mkdir build && cd build
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=~/.vcpkg/scripts/buildsystems/vcpkg.cmake -DENABLE_TESTS=OFF -DBUILD_SERVING_BIN=OFF
make -j8
```

After the compilation is complete, metaspore-1.0.1-cp38-cp38-linux_x86_64.whl will be generated in the current build directory, where the version `1.0.1` is read from pyproject.toml file.

Then you can install it into the Python environment for testing with the following command:
```bash
pip install --upgrade --force-reinstall --no-deps metaspore-1.0.1-cp38-cp38-linux_x86_64.whl
```