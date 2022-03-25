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

#include <common/hashmap/memory_mapped_array_hash_map.h>
#include <common/hashmap/memory_mapped_array_hash_map_loader.h>
#include <ctype.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace metaspore;

template <typename TKey, typename TValue>
void dump_map(const MemoryMappedArrayHashMapLoader &loader, uint64_t num_dump_keys) {
    if (sizeof(TKey) != sizeof(uint64_t))
        throw std::runtime_error("sizeof(uint64_t) key type expected");
    MemoryMappedArrayHashMap<TKey, TValue> map = loader.get<TKey, TValue>();
    std::cout << "hash_code: " << map.get_hash_code() << std::endl;
    std::cout << "data:" << std::endl;
    std::cout << "----------" << std::endl;
    uint64_t index = 0;
    for (TKey key : map) {
        if (index++ >= num_dump_keys)
            break;
        uint64_t count;
        const TValue *values = map.get(key, count);
        if (sizeof(TValue) * count % sizeof(float) != 0) {
            std::ostringstream serr;
            auto dtype = DataTypeToCode<TValue>::value;
            serr << count << " " << DataTypeToString(dtype);
            serr << " can not be cast as float array.";
            throw std::runtime_error(serr.str());
        }
        const float *fvalues = (const float *)values;
        const uint64_t fcount = sizeof(TValue) * count / sizeof(float);
        std::cout << static_cast<uint64_t>(key) << ": [";
        for (uint64_t j = 0; j < fcount; j++)
            std::cout << (j ? ", " : "") << fvalues[j];
        std::cout << "]" << std::endl;
    }
}

template <typename TKey>
void dump_map(const MemoryMappedArrayHashMapLoader &loader, DataType value_type,
              uint64_t num_dump_keys) {
    switch (value_type) {
#undef MS_DATA_STRUCTURES_DATA_TYPE_DEF
#define MS_DATA_STRUCTURES_DATA_TYPE_DEF(t, l, u)                                                  \
    case DataType::u:                                                                              \
        dump_map<TKey, t>(loader, num_dump_keys);                                                  \
        break;                                                                                     \
        /**/
        MS_DATA_STRUCTURES_DATA_TYPES(MS_DATA_STRUCTURES_DATA_TYPE_DEF)
    }
}

void dump_map(const MemoryMappedArrayHashMapLoader &loader, DataType key_type, DataType value_type,
              uint64_t num_dump_keys) {
    switch (key_type) {
#undef MS_DATA_STRUCTURES_DATA_TYPE_DEF
#define MS_DATA_STRUCTURES_DATA_TYPE_DEF(t, l, u)                                                  \
    case DataType::u:                                                                              \
        dump_map<t>(loader, value_type, num_dump_keys);                                            \
        break;                                                                                     \
        /**/
        MS_DATA_STRUCTURES_DATA_TYPES(MS_DATA_STRUCTURES_DATA_TYPE_DEF)
    }
}

char hex_to_string(int digit) {
    if (digit <= 9)
        return static_cast<char>('0' + digit);
    else
        return static_cast<char>('A' + digit - 10);
}

std::string signature_to_string(const char *sig, size_t len) {
    std::string buf;
    for (size_t i = 0; i < len; i++) {
        if (isgraph(sig[i])) {
            if (i > 0 && !isgraph(sig[i - 1]))
                buf += ' ';
            buf += sig[i];
        } else {
            const int b = static_cast<unsigned char>(sig[i]);
            if (i > 0)
                buf += ' ';
            buf += hex_to_string((b >> 4) & 0xF);
            buf += hex_to_string(b & 0xF);
        }
    }
    return buf;
}

void dump_map_file(const std::string &input_path, uint64_t num_dump_keys) {
    std::cout << "input_path: " << input_path << std::endl;
    std::cout << "num_dump_keys: ";
    if (num_dump_keys == static_cast<uint64_t>(-1))
        std::cout << "all";
    else
        std::cout << num_dump_keys;
    std::cout << std::endl;
    MemoryMappedArrayHashMapLoader loader(input_path);
    const MapFileHeader &header = loader.get_header();
    std::cout << "signature: " << signature_to_string(header.signature, sizeof(header.signature))
              << std::endl;
    std::cout << "version: " << header.version << std::endl;
    std::cout << "is_optimized_mode: " << header.is_optimized_mode << std::endl;
    const DataType key_type = static_cast<DataType>(header.key_type);
    const DataType value_type = static_cast<DataType>(header.value_type);
    std::cout << "key_type: " << DataTypeToString(key_type) << std::endl;
    std::cout << "value_type: " << DataTypeToString(value_type) << std::endl;
    std::cout << "key_count: " << header.key_count << std::endl;
    std::cout << "bucket_count: " << header.bucket_count << std::endl;
    std::cout << "value_count: " << header.value_count << std::endl;
    std::cout << "value_count_per_key: " << header.value_count_per_key << std::endl;
    dump_map(loader, key_type, value_type, num_dump_keys);
}

int main(int argc, char **argv) {
    if (argc <= 1) {
        std::cout << "usage:\n\n";
        std::cout << "  " << argv[0] << " map_file_path [num_dump_keys]\n";
        std::cout << "\nexample:\n\n";
        std::cout << "  1. load map.dat and dump at most 10 keys\n\n";
        std::cout << "     " << argv[0] << " map.dat\n\n";
        std::cout << "  2. load map.dat and dump at most 100 keys\n\n";
        std::cout << "     " << argv[0] << " map.dat 100\n\n";
        std::cout << "  3. load big_map.dat and dump all keys\n\n";
        std::cout << "     " << argv[0] << " big_map.dat 0\n\n";
        return 0;
    }
    long num = 0;
    if (argc >= 3)
        num = std::stol(argv[2]);
    else
        num = 10;
    const std::string input_path = argv[1];
    const uint64_t num_dump_keys = static_cast<uint64_t>(num <= 0 ? -1 : num);
    try {
        dump_map_file(input_path, num_dump_keys);
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
