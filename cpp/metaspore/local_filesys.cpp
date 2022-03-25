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

#include "common/logger.h"
#include <metaspore/local_filesys.h>

#include <filesystem>
#include <fstream>

#include <boost/algorithm/string.hpp>
#include <fmt/format.h>

namespace metaspore {

namespace fs = std::filesystem;

/*! \brief implementation of file i/o stream */
template <typename FStream> class FileStream : public SeekStream {
  public:
    explicit FileStream(FStream &&_fs) : fs_(std::move(_fs)) {}

    virtual ~FileStream(void) { this->Close(); }

    virtual bool AtEnd(void) const { return fs_.eof(); }

    inline void Close(void) { fs_.close(); }

    virtual size_t Read(void *ptr, size_t size) override {
        throw std::runtime_error("Read not implemented for this stream");
    }

    virtual void Write(const void *ptr, size_t size) override {
        throw std::runtime_error("Write not implemented for this stream");
    }

  protected:
    FStream fs_;
};

class FileInputStream : public FileStream<std::ifstream> {
  public:
    using FileStream<std::ifstream>::FileStream;

    void Seek(size_t pos) override {
        if (!fs_.seekg(pos)) {
            throw std::runtime_error(fmt::format("Seek to {} failed for local file input stream {}",
                                                 pos, std::strerror(errno)));
        }
    }

    size_t Tell(void) override { return fs_.tellg(); }

    size_t Read(void *ptr, size_t size) override {
        if (!fs_.read((char *)ptr, size)) {
            if (fs_.eof()) {
                return fs_.gcount();
            }
            throw std::runtime_error(
                fmt::format("Read size {} failed for local file input stream {}", size, std::strerror(errno)));
        }
        return size;
    }
};

class FileOutputStream : public FileStream<std::ofstream> {
  public:
    using FileStream<std::ofstream>::FileStream;

    void Seek(size_t pos) override {
        if (!fs_.seekp(pos)) {
            throw std::runtime_error(fmt::format(
                "Seek to {} failed for local file output stream {}", pos, std::strerror(errno)));
        }
    }

    size_t Tell(void) override { return fs_.tellp(); }

    void Write(const void *ptr, size_t size) override {
        if (!fs_.write((const char *)ptr, size)) {
            throw std::runtime_error(
                fmt::format("Write failed for local file output stream {}", std::strerror(errno)));
        }
    }
};

FileInfo LocalFileSystem::GetPathInfo(const URI &path) {
    FileInfo ret;
    ret.path = path;
    fs::path p{path.name};
    if (fs::is_directory(p)) {
        ret.type = kDirectory;
    } else if (fs::is_regular_file(p)) {
        ret.type = kFile;
        ret.size = fs::file_size(p);
    }
    SPDLOG_INFO("Get Path Info for {} with size {}", path.name, ret.size);
    return ret;
}

void LocalFileSystem::ListDirectory(const URI &path, std::vector<FileInfo> *out_list) {}

SeekStream *LocalFileSystem::Open(const URI &path, const char *const mode, bool allow_null) {
    SPDLOG_INFO("Opening local file {} with mode {}", path.name, mode);
    std::string m(mode);
    if (boost::contains(m, "r")) {
        std::ifstream ifs;
        auto mode = std::ios_base::in;
        if (boost::contains(m, "b")) {
            mode |= std::ios_base::binary;
        }
        ifs.open(path.name.c_str(), mode);
        if (!ifs.is_open()) {
            if (allow_null) {
                return nullptr;
            } else {
                throw std::runtime_error(fmt::format("Open local file {} failed with mode {}: {}",
                                                     path.name, mode, std::strerror(errno)));
            }
        }
        return new FileInputStream(std::move(ifs));
    } else if (boost::contains(m, "w")) {
        std::ofstream ofs;
        auto mode = std::ios_base::out;
        if (boost::contains(m, "b")) {
            mode |= std::ios_base::binary;
        }
        ofs.open(path.name.c_str(), mode);
        if (!ofs.is_open()) {
            if (allow_null) {
                return nullptr;
            } else {
                throw std::runtime_error(fmt::format("Open local file {} failed with mode {}: {}",
                                                     path.name, mode, std::strerror(errno)));
            }
        }
        return new FileOutputStream(std::move(ofs));
    }
    if (allow_null)
        return nullptr;
    throw std::runtime_error(
        fmt::format("Cannot open local stream {} with mode {}", path.name, mode));
}

SeekStream *LocalFileSystem::OpenForRead(const URI &path, bool allow_null) {
    return Open(path, "r", allow_null);
}

} // namespace metaspore
