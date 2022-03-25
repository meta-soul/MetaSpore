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

#include "metaspore/io.h"
#include "common/logger.h"
#include "metaspore/filesys.h"
#include "metaspore/local_filesys.h"
#include "metaspore/s3_sdk_filesys.h"
#include "metaspore/stack_trace_utils.h"
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/types.h>

namespace metaspore {

FileSystem *FileSystem::GetInstance(const URI &path) {
    if (path.protocol == "file://" || path.protocol.length() == 0) {
        return LocalFileSystem::GetInstance();
    }
    if (path.protocol == "hdfs://" || path.protocol == "viewfs://") {
#if DMLC_USE_HDFS
        if (path.host.length() == 0) {
            return HDFSFileSystem::GetInstance("default");
        } else if (path.protocol == "viewfs://") {
            char *defaultFS = nullptr;
            hdfsConfGetStr("fs.defaultFS", &defaultFS);
            if (path.host.length() != 0) {
                CHECK("viewfs://" + path.host == defaultFS)
                    << "viewfs is only supported as a fs.defaultFS.";
            }
            return HDFSFileSystem::GetInstance("default");
        } else {
            return HDFSFileSystem::GetInstance(path.host);
        }
#else
        SPDLOG_CRITICAL("Please compile with DMLC_USE_HDFS=1 to use hdfs");
#endif
    }
    if (path.protocol == "s3://" || path.protocol == "http://" || path.protocol == "https://") {
#if DMLC_USE_S3
        return S3FileSystem::GetInstance();
#else
        LOG(FATAL) << "Please compile with DMLC_USE_S3=1 to use S3";
#endif
    }

    SPDLOG_CRITICAL("unknown filesystem protocol {}", path.protocol);
    return NULL;
}
Stream *Stream::Create(const char *uri, const char *const flag, bool try_create) {
    URI path(uri);
    return FileSystem::GetInstance(path)->Open(path, flag, try_create);
}

SeekStream *SeekStream::CreateForRead(const char *uri, bool try_create) {
    URI path(uri);
    return FileSystem::GetInstance(path)->OpenForRead(path, try_create);
}

InputStream::InputStream(const std::string &url) : stream_(Stream::Create(url.c_str(), "r", true)) {
    if (!stream_) {
        std::string serr;
        serr.append("Fail to open '");
        serr.append(url);
        serr.append("' for input.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
}

size_t InputStream::Read(void *buffer, size_t size) { return stream_->Read(buffer, size); }

OutputStream::OutputStream(const std::string &url)
    : stream_(Stream::Create(url.c_str(), "w", true)) {
    if (!stream_) {
        std::string serr;
        serr.append("Fail to open '");
        serr.append(url);
        serr.append("' for output.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
}

void OutputStream::Write(const void *buffer, size_t size) { stream_->Write(buffer, size); }

void StreamWriteAll(const std::string &url, const char *data, size_t size) {
    auto stream = Stream::Create(url.c_str(), "w", true);
    if (!stream) {
        std::string serr;
        serr.append("Fail to open '");
        serr.append(url);
        serr.append("' for writing.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    std::unique_ptr<Stream> stream_guard(stream);
    stream->Write(data, size);
}

void StreamWriteAll(const std::string &url, const std::string &data) {
    StreamWriteAll(url, data.data(), data.size());
}

void StreamReadAll(const std::string &url, char *data, size_t size) {
    auto stream = Stream::Create(url.c_str(), "r", true);
    if (!stream) {
        std::string serr;
        serr.append("Fail to open '");
        serr.append(url);
        serr.append("' for reading.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    std::unique_ptr<Stream> stream_guard(stream);
    const size_t nread = stream->Read(data, size);
    if (nread != size) {
        std::string serr;
        serr.append("Try to read ");
        serr.append(std::to_string(size));
        serr.append(" bytes from '");
        serr.append(url);
        serr.append("', but only ");
        serr.append(std::to_string(nread));
        serr.append(" bytes are read successfully.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
}

std::string StreamReadAll(const std::string &url) {
    auto stream = Stream::Create(url.c_str(), "r", true);
    if (!stream) {
        std::string serr;
        serr.append("Fail to open '");
        serr.append(url);
        serr.append("' for reading.\n\n");
        serr.append(GetStackTrace());
        spdlog::error(serr);
        throw std::runtime_error(serr);
    }
    std::unique_ptr<Stream> stream_guard(stream);
    std::string result;
    std::string buffer(1024 * 1024, '\0');
    size_t n;
    while ((n = stream->Read(&buffer.front(), buffer.size())) != 0)
        result.append(&buffer.front(), n);
    return result;
}

void MakeLocalDirectories(const std::string &path, mode_t mode) {
    if (path.empty())
        return;
    std::string buf = path;
    char *dir_path = const_cast<char *>(buf.c_str());
    char *p = dir_path;
    do {
        p = strchr(++p, '/');
        if (p)
            *p = '\0';
        if (mkdir(dir_path, mode) == -1 && errno != EEXIST) {
            std::string serr;
            serr.append("Fail to make directory '");
            serr.append(dir_path);
            serr.append("'. errno [");
            serr.append(std::to_string(errno));
            serr.append("]: ");
            serr.append(strerror(errno));
            serr.append("\n\n");
            serr.append(GetStackTrace());
            spdlog::error(serr);
            throw std::runtime_error(serr);
        }
        if (p)
            *p = '/';
    } while (p);
}

void EnsureLocalDirectory(const std::string &dir_path) {
    URI uri(dir_path.c_str());
    if (uri.protocol.empty() || uri.protocol == "file://")
        MakeLocalDirectories(uri.name, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
}

std::string DirName(const std::string &path) {
    size_t i = path.rfind('/');
    if (i == std::string::npos)
        return {};
    else
        return path.substr(0, i);
}

std::string JoinPath(const std::string &dir_path, const std::string &file_name) {
    std::string path = dir_path;
    if (path.empty())
        path = "./";
    else if (path.at(path.size() - 1) != '/' && path.at(path.size() - 1) != '\\')
        path += "/";
    path += file_name;
    return path;
}

} // namespace metaspore
