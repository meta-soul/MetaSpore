# 离线训练 Python Wheel 包构建

## 准备环境
推荐使用 Ubuntu 20.04 作为开发环境。MetaSpore 框架采用 C++ 20 开发，并提供了 Python 接口。

首先安装开发依赖：
```bash
sudo apt install build-essential manpages-dev software-properties-common curl zip unzip tar pkg-config bison flex
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt install gcc-11 g++-11
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 --slave /usr/bin/g++ g++ /usr/bin/g++-9 --slave /usr/bin/gcov gcov /usr/bin/gcov-9 --slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-9 --slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-9
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110 --slave /usr/bin/g++ g++ /usr/bin/g++-11 --slave /usr/bin/gcov gcov /usr/bin/gcov-11 --slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-11 --slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-11
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null 
sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" 
sudo apt update
sudo apt install cmake
git clone https://github.com/Microsoft/vcpkg.git ~/.vcpkg
~/.vcpkg/bootstrap-vcpkg.sh
```

## 编译代码
```bash
git clone https://github.com/meta-soul/MetaSpore.git
cd MetaSpore
mkdir build && cd build
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=~/.vcpkg/scripts/buildsystems/vcpkg.cmake -DENABLE_TESTS=OFF -DBUILD_SERVING_BIN=OFF
make -j8
```
编译完成后会在当前 build 目录生成 metaspore-1.0.0-GITTAG-cp38-cp38-linux_x86_64.whl，其中 GITTAG 需要替换为实际的 git 版本号。

然后可以通过如下命令安装到 Python 环境进行测试：
```bash
pip install --upgrade --force-reinstall --no-deps metaspore-1.0.0+GITTAG-cp38-cp38-linux_x86_64.whl
```