//
// Copyright 2022 DMetaSoul
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include <serving/ort_model.h>
#include <common/test_utils.h>
#include <boost/asio/use_future.hpp>

#include <fstream>
#include <thread>

using namespace metaspore;
using namespace metaspore::serving;
using namespace std::chrono_literals;
using namespace std::string_literals;

static Ort::AllocatorWithDefaultOptions alloc;

// modified from
// https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/src/data/datasets/mnist.cpp
bool check_is_little_endian() {
    const uint32_t word = 1;
    return reinterpret_cast<const uint8_t *>(&word)[0] == 1;
}

constexpr uint32_t flip_endianness(uint32_t value) {
    return ((value & 0xffu) << 24u) | ((value & 0xff00u) << 8u) | ((value & 0xff0000u) >> 8u) |
           ((value & 0xff000000u) >> 24u);
}

uint32_t read_int32(std::ifstream &stream) {
    static const bool is_little_endian = check_is_little_endian();
    uint32_t value;
    stream.read(reinterpret_cast<char *>(&value), sizeof value);
    return is_little_endian ? flip_endianness(value) : value;
}

template <typename T, typename R> void cast_to(const T *from, R *to, size_t num) {
    for (size_t i = 0; i < num; ++i) {
        to[i] = static_cast<R>(from[i]);
    }
}

template <typename T> void normalize(T *p, T d, size_t num) {
    for (size_t i = 0; i < num; ++i) {
        p[i] /= d;
    }
}

Ort::Value get_mnist_test_10_images_tensor(const std::string &path, int64_t batch_size) {
    std::ifstream images(path, std::ios::binary);
    auto magic = read_int32(images);
    auto count = read_int32(images);
    auto rows = read_int32(images);
    EXPECT_EQ(rows, 28);
    auto cols = read_int32(images);
    EXPECT_EQ(cols, 28);
    std::array<int64_t, 4> shape{batch_size, 1L, rows, cols};
    auto value = Ort::Value::CreateTensor<float>(alloc, &shape[0], 4);
    size_t numelem = batch_size * rows * cols;
    auto buffer = std::make_unique<uint8_t[]>(numelem);
    images.read((char *)buffer.get(), numelem);
    cast_to(buffer.get(), value.GetTensorMutableData<float>(), numelem);
    normalize(value.GetTensorMutableData<float>(), float(255), numelem);
    return value;
}

#define EXPECT_TRUE_COROUTINE(status)                                                              \
    EXPECT_TRUE(status.ok()) << status;                                                            \
    if (!status.ok()) {                                                                            \
        co_return;                                                                                 \
    }

TEST(ORT_MODEL_TEST_SUITE, TestOrtModelLoadNormal) {
    auto &tp = Threadpools::get_background_threadpool();
    std::future<void> future = boost::asio::co_spawn(
        tp,
        []() -> awaitable<void> {
            OrtModel model;
            auto status = co_await model.load("mnist_model");
            EXPECT_TRUE_COROUTINE(status);
            auto info = model.info();
            EXPECT_EQ(info, "onnxruntime model loaded from mnist_model/model.onnx"s);
            auto input_value = get_mnist_test_10_images_tensor("data/MNIST/raw/t10k-images-idx3-ubyte"s, 10);
            const float *in_ptr = input_value.GetTensorMutableData<float>();
            auto input = std::make_unique<OrtModelInput>();
            input->inputs.emplace("input"s, OrtModelInput::Value{.value = std::move(input_value)});
            auto result = co_await model.predict(std::move(input));
            EXPECT_TRUE_COROUTINE(result.status());
            OrtModelOutput *output = dynamic_cast<OrtModelOutput *>(result->get());
            EXPECT_NE(output, nullptr);
            auto find_output = output->outputs.find("output"s);
            EXPECT_NE(find_output, output->outputs.end());
            Ort::Value output_value = std::move(find_output->second);
            TensorPrint::print_tensor<float>(output_value);
            co_return;
        },
        boost::asio::use_future);
    future.get();
}

int main(int argc, char **argv) { return run_all_tests(argc, argv); }
