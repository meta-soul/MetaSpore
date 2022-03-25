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

#include <common/hashmap/map_file_header.h>
#include <common/hashmap/memory_mapped_array_hash_map_loader.h>
#include <errno.h>
#include <fcntl.h>
#include <sstream>
#include <stdexcept>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace metaspore {

MemoryMappedArrayHashMapLoader::MemoryMappedArrayHashMapLoader(const std::string &path,
                                                               bool disableMmap) {
    std::string hint;
    hint.append("Fail to load map from \"");
    hint.append(path);
    hint.append("\"; ");
    int fd = open(path.c_str(), 0);
    if (fd == -1) {
        std::ostringstream serr;
        serr << hint;
        serr << "can not open file. ";
        serr << strerror(errno);
        throw std::runtime_error(serr.str());
    }
    auto deleter = [](int *pfd) { close(*pfd); };
    std::unique_ptr<int, decltype(deleter)> fd_guard(&fd, deleter);
    struct stat st;
    if (fstat(fd, &st) == -1) {
        std::ostringstream serr;
        serr << hint;
        serr << "can not stat to get file size. ";
        serr << strerror(errno);
        throw std::runtime_error(serr.str());
    }
    const uint64_t size = static_cast<uint64_t>(st.st_size);
    if (size < map_file_header_size) {
        std::ostringstream serr;
        serr << hint;
        serr << "file is too small to contain a header. ";
        throw std::runtime_error(serr.str());
    }
    std::shared_ptr<void> ptr_guard;
    spdlog::info("MemoryMappedArrayHashMapLoader using {} mode.",
                 (disableMmap ? "malloc" : "mmap"));
    if (disableMmap) {
        void *ptr = malloc(size);
        if (ptr == NULL) {
            std::ostringstream serr;
            serr << hint;
            serr << "can not malloc " << size << " bytes. ";
            serr << strerror(errno);
            throw std::runtime_error(serr.str());
        }
        ptr_guard = std::shared_ptr<void>(ptr, free);
        ssize_t ret = read(fd, ptr, size);
        if (ret != size) {
            std::ostringstream serr;
            serr << hint;
            serr << "read file failed, size = " << size << ", ret = " << ret << ". ";
            serr << strerror(errno);
            throw std::runtime_error(serr.str());
        }
    } else {
        void *ptr = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (ptr == MAP_FAILED) {
            std::ostringstream serr;
            serr << hint;
            serr << "can not mmap file. ";
            serr << strerror(errno);
            throw std::runtime_error(serr.str());
        }
        auto unmapper = [size](void *ptr) { munmap(ptr, size); };
        ptr_guard = std::shared_ptr<void>(ptr, unmapper);
    }
    fd_guard.reset();
    const MapFileHeader &header = *static_cast<const MapFileHeader *>(ptr_guard.get());
    if (header.is_optimized_mode == 1)
        header.validate(true, hint);
    path_ = path;
    blob_ = std::move(ptr_guard);
    size_ = size;
}

} // namespace metaspore