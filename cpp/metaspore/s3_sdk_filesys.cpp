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

#include <metaspore/s3_sdk_filesys.h>

#include <cstdlib>
#include <fstream>
#include <iostream>

#include <aws/core/Aws.h>
#include <aws/core/auth/AWSCredentialsProvider.h>
#include <aws/core/client/ClientConfiguration.h>
#include <aws/core/client/DefaultRetryStrategy.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/AbortMultipartUploadRequest.h>
#include <aws/s3/model/CompleteMultipartUploadRequest.h>
#include <aws/s3/model/CompleteMultipartUploadResult.h>
#include <aws/s3/model/CompletedMultipartUpload.h>
#include <aws/s3/model/CompletedPart.h>
#include <aws/s3/model/CreateMultipartUploadRequest.h>
#include <aws/s3/model/CreateMultipartUploadResult.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/HeadObjectRequest.h>
#include <aws/s3/model/HeadObjectResult.h>
#include <aws/s3/model/ListObjectsRequest.h>
#include <aws/s3/model/Object.h>
#include <aws/s3/model/PutObjectRequest.h>
#include <aws/s3/model/UploadPartRequest.h>
#include <aws/s3/model/UploadPartResult.h>

#include <aws/core/utils/logging/AWSLogging.h>
#include <aws/core/utils/logging/ConsoleLogSystem.h>
#include <aws/core/utils/memory/stl/AWSString.h>

#include "common/logger.h"

namespace metaspore {

class AWSInitOption {
  public:
    AWSInitOption() {
        Aws::InitAPI(options);
        clientConfigPtr = std::make_shared<Aws::Client::ClientConfiguration>();
        const char *ep = getenv("AWS_ENDPOINT");
        if (ep) {
            std::cout << "Get aws endpoint from env: " << ep << std::endl;
            clientConfigPtr->endpointOverride = ep;
        }
        clientConfigPtr->scheme = Aws::Http::Scheme::HTTP;
        clientConfigPtr->connectTimeoutMs = 60000;
        clientConfigPtr->requestTimeoutMs = 60000;
        std::shared_ptr<Aws::Client::RetryStrategy> retry;
        retry.reset(new Aws::Client::DefaultRetryStrategy(10, 5));
        clientConfigPtr->retryStrategy = retry; // assign to client_config

        Aws::Utils::Logging::InitializeAWSLogging(
            Aws::MakeShared<Aws::Utils::Logging::ConsoleLogSystem>(
                "S3Logging", Aws::Utils::Logging::LogLevel::Warn));
    }

    ~AWSInitOption() {
        clientConfigPtr.reset();
        Aws::ShutdownAPI(options);
    }

    Aws::SDKOptions options;
    std::shared_ptr<Aws::Client::ClientConfiguration> clientConfigPtr;

    static AWSInitOption &GetInstance() {
        static AWSInitOption option;
        return option;
    }
};

static const char *GetValidKey(const char *key, size_t len) {
    if (len == 0) {
        return key;
    }
    for (size_t i = 0; i < len - 1; ++i) {
        if (key[i] != '/') {
            return key + i;
        }
    }
    return key;
}

static Aws::S3::Model::ListObjectsOutcome ListS3Objects(const URI &path) {
    Aws::S3::S3Client s3_client(*AWSInitOption::GetInstance().clientConfigPtr);
    Aws::S3::Model::ListObjectsRequest objects_request;

    const char *prefix = GetValidKey(path.name.c_str(), path.name.size());
    SPDLOG_INFO("S3FileSystem::ListDirectory {}, {}, {}", path.protocol, path.host, prefix);

    objects_request.WithBucket(path.host.c_str()).SetPrefix(prefix);

    auto list_objects_outcome = s3_client.ListObjects(objects_request);

    if (list_objects_outcome.IsSuccess()) {
        return std::move(list_objects_outcome);
    } else {
        SPDLOG_ERROR("ListObjects error: {}, {}",
                     list_objects_outcome.GetError().GetExceptionName(),
                     list_objects_outcome.GetError().GetMessage());
        return Aws::S3::Model::ListObjectsOutcome();
    }
}

/*!
 * \brief get information about a path
 * \param path the path to the file
 * \return the information about the file
 */
FileInfo S3FileSystem::GetPathInfo(const URI &path) {
    auto list_objects_outcome = ListS3Objects(path);
    FileInfo info;
    info.path = path;
    const Aws::Vector<Aws::S3::Model::Object> &object_list =
        list_objects_outcome.GetResult().GetContents();

    if (object_list.size() > 1) {
        info.type = kDirectory;
        info.size = 0UL;
    } else {
        info.type = kFile;
        auto const &s3_object = object_list.front();
        info.size = s3_object.GetSize();
    }
    return std::move(info);
}

/*!
 * \brief list files in a directory
 * \param path to the file
 * \param out_list the output information about the files
 */
void S3FileSystem::ListDirectory(const URI &path, std::vector<FileInfo> *out_list) {
    auto list_objects_outcome = ListS3Objects(path);
    const Aws::Vector<Aws::S3::Model::Object> &object_list =
        list_objects_outcome.GetResult().GetContents();
    for (auto const &object : object_list) {
        auto const &key = object.GetKey();
        out_list->emplace_back();
        FileInfo &info = out_list->back();
        info.path = path;
        if (key.front() != '/') {
            info.path.name = '/';
            info.path.name += key.c_str();
        } else {
            info.path.name = key.c_str();
        }
        info.size = key.back() == '/' ? 0 : object.GetSize();
        info.type = key.back() == '/' ? kDirectory : kFile;
    }
}

class ReadBuffer {
  public:
    ReadBuffer() : buf_(), pos_(), size_() {}
    void Init() { buf_.reserve(read_buf_size); }
    size_t LeftRoom() const { return size_ - pos_; }
    void Clear() {
        buf_.clear();
        pos_ = 0;
        size_ = 0;
    }
    size_t FillBuffer(Stream *s);
    void UseBuffer(void *ptr, size_t len) {
        // return buf_[pos_] + len
        memcpy(ptr, &buf_[pos_], len);
        pos_ += len;
    }
    size_t Read(Stream *s, void *ptr, size_t len) {
        const size_t left = LeftRoom();

        // current left room is enough for this read request
        if (left >= len) {
            UseBuffer(ptr, len);
            return len;
        }

        // not enough room.
        // 1. copy left root
        if (left > 0) {
            UseBuffer(ptr, left);
        }
        // 2. refill buffer until done
        size_t stillneed = len - left;
        while (stillneed > 0) {
            const size_t nread = FillBuffer(s);
            if (nread == 0) {
                // no more data
                return len - stillneed;
            }
            if (nread >= stillneed) {
                // finished reading
                UseBuffer((char *)ptr + len - stillneed, stillneed);
                return len;
            } else {
                UseBuffer((char *)ptr + len - stillneed, nread);
                stillneed -= nread;
                // continue loop
            }
        }
        return (len - stillneed);
    }

  private:
    std::string buf_;
    size_t pos_;  // current used pos
    size_t size_; // valid size of current buffer
    // read buffer size for prefetch
    static const size_t read_buf_size = 1024UL * 1024UL * 8UL;
};

class WriteBuffer {
  public:
    WriteBuffer(Aws::S3::S3Client &client) : client_(client) {}
    void Init(Aws::String &bucket, Aws::String &key) {
        buf_.reserve(write_buf_size);
        // Initiate upload part
        bucket_ = bucket;
        key_ = key;
        Aws::S3::Model::CreateMultipartUploadRequest request;
        request.WithBucket(bucket_).WithKey(key_);
        auto const createUploadOutcome = client_.CreateMultipartUpload(request);
        if (createUploadOutcome.IsSuccess()) {
            auto const &result = createUploadOutcome.GetResult();
            upload_id_ = result.GetUploadId();
        } else {
            SPDLOG_ERROR("CreateMultipartUploadRequest error for file: s3://{}/{}, {} {}", bucket_,
                         key_, createUploadOutcome.GetError().GetExceptionName(),
                         createUploadOutcome.GetError().GetMessage());
            throw std::runtime_error("CreateMultipartUpload error");
        }
    }

    void Write(const void *ptr, size_t len) {
        // append to buffer
        buf_.append((const char *)ptr, len);
        if (buf_.length() >= write_buf_size) {
            DoUploadPart();
            buf_.clear();
        }
    }

    template <typename Request> void FillRequest(Request &request) {
        request.WithBucket(bucket_).WithKey(key_).WithContentLength(buf_.size());
        request.SetContentType("binary/octet-stream");
        auto s = Aws::MakeShared<Aws::StringStream>("WriteObjectStream",
                                                    std::stringstream::in | std::stringstream::out |
                                                        std::stringstream::binary);
        s->write(buf_.c_str(), buf_.size());
        request.SetBody(s);
    }

    void DoPutObject() {
        Aws::S3::Model::PutObjectRequest request;
        FillRequest(request);
        auto const outcome = client_.PutObject(request);
        if (!outcome.IsSuccess()) {
            SPDLOG_ERROR("PutObjectRequest error for file: s3://{}/{}, {} {}", bucket_, key_,
                         outcome.GetError().GetExceptionName(), outcome.GetError().GetMessage());
            throw std::runtime_error("PutObjectRequest error");
        }
    }

    void DoUploadPart() {
        Aws::S3::Model::UploadPartRequest request;
        const int partNum = static_cast<int>(parts_.GetParts().size() + 1);
        FillRequest(request);
        request.WithPartNumber(partNum).WithUploadId(upload_id_).WithContentLength(buf_.size());

        auto const outcome = client_.UploadPart(request);
        if (!outcome.IsSuccess()) {
            SPDLOG_ERROR("UploadPart error for file: s3://{}/{}, {} {}", bucket_, key_,
                         outcome.GetError().GetExceptionName(), outcome.GetError().GetMessage());
            throw std::runtime_error("UploadPart error");
        }
        Aws::S3::Model::CompletedPart part;
        part.SetETag(outcome.GetResult().GetETag());
        part.SetPartNumber(partNum);
        parts_.AddParts(std::move(part));
    }

    void Close() {
        if (!buf_.empty()) {
            if (parts_.GetParts().empty()) {
                // no previous part, directly put
                DoPutObject();
            } else {
                // upload final part
                DoUploadPart();
            }
        }
        if (!parts_.GetParts().empty()) {
            // complete part upload
            Aws::S3::Model::CompleteMultipartUploadRequest request;
            request.WithBucket(bucket_)
                .WithKey(key_)
                .WithUploadId(upload_id_)
                .WithMultipartUpload(parts_);
            auto const complete_outcome = client_.CompleteMultipartUpload(request);
            if (!complete_outcome.IsSuccess()) {
                SPDLOG_ERROR("CompleteMultipartUpload error for file: s3://{}/{}, {} {}", bucket_,
                             key_, complete_outcome.GetError().GetExceptionName(),
                             complete_outcome.GetError().GetMessage());
                throw std::runtime_error("CompleteMultipartUpload error");
            }
        } else {
            // abort part upload
            Aws::S3::Model::AbortMultipartUploadRequest request;
            request.WithBucket(bucket_).WithKey(key_).WithUploadId(upload_id_);
            client_.AbortMultipartUpload(request);
        }
    }

  private:
    Aws::S3::S3Client &client_;
    std::string buf_;
    Aws::String bucket_;
    Aws::String key_;
    Aws::String upload_id_;
    Aws::S3::Model::CompletedMultipartUpload parts_;
    // default buffer size for caching write
    static const size_t write_buf_size = 1024UL * 1024UL * 5UL;
};

class S3SDKStream : public SeekStream {
  public:
    S3SDKStream()
        : client_(*AWSInitOption::GetInstance().clientConfigPtr), pos_(), size_(),
          write_buf_(client_), is_write_(false) {}

    virtual ~S3SDKStream() {
        if (is_write_) {
            write_buf_.Close();
        }
    }

    bool Open(const URI &path, bool read_only) {
        path_ = path;
        bucket_ = path.host.c_str();
        key_ = GetValidKey(path.name.c_str(), path.name.size());
        SPDLOG_INFO("Try to open S3 stream: s3://{}/{}, read_only {}", bucket_, key_, read_only);
        if (path_.name.back() == '/') {
            SPDLOG_ERROR("S3 open stream with a directory path: {}", path.name);
            throw std::runtime_error(path.name + " is not a valid file path");
        }

        if (read_only) {
            Aws::S3::Model::HeadObjectRequest object_request;

            object_request.WithBucket(bucket_).WithKey(key_);

            auto const head_object_outcome = client_.HeadObject(object_request);

            if (head_object_outcome.IsSuccess()) {
                const int64_t length = head_object_outcome.GetResult().GetContentLength();

                if (length < 0) {
                    SPDLOG_ERROR(
                        "Open read-only stream for object: s3://{}/{} but with invalid length: {}",
                        bucket_, key_, length);
                    return false;
                }

                SPDLOG_INFO("Opened read-only stream for object: s3://{}/{} with total length {}",
                            bucket_, key_, length);
                size_ = length;
                read_buf_.Init(); // init read prefetch buffer
                return true;
            } else {
                SPDLOG_INFO("Read object s3://{}/{} failed with error: {}", bucket_, key_,
                            head_object_outcome.GetError().GetMessage());
                throw std::runtime_error(std::string("Read object s3://" + bucket_ + "/" + key_ +
                                                     " failed with error: ") +
                                         head_object_outcome.GetError().GetMessage());
            }
        } else {
            is_write_ = true;
            write_buf_.Init(bucket_, key_);
        }

        return false;
    }

    virtual size_t Read(void *ptr, size_t size) override {
        // try prefetch
        const size_t rlen = read_buf_.Read(this, ptr, size);
        return rlen;
    }

    /**
     *  For reading, aws sdk actually performs new http request for each
     *  range, so there is no need to maintain a local stream.
     *  Upper layer user should use proper buffer size to optimize performance.
     */
    size_t ActualRead(void *ptr, size_t size) {
        if (pos_ == size_) {
            SPDLOG_INFO("Read S3 object s3://{}/{} reached end {}", bucket_, key_, pos_);
            return 0UL;
        }

        if (pos_ + size > size_) {
            size = size_ - pos_;
            SPDLOG_INFO("Read S3 object s3://{}/{} with size {} at position {} larger than total "
                        "size: {}, change size to {}",
                        bucket_, key_, size, pos_, size_, size);
        }

        // requesting range
        Aws::S3::Model::GetObjectRequest object_request;
        object_request.WithBucket(bucket_).WithKey(key_).WithRange(
            ("bytes=" + std::to_string(pos_) + "-" + std::to_string(pos_ + size)).c_str());

        auto get_object_outcome = client_.GetObject(object_request);

        if (get_object_outcome.IsSuccess()) {
            Aws::IOStream &input_stream = get_object_outcome.GetResult().GetBody();
            input_stream.read((char *)ptr, size);
            pos_ += size;
            return size;
        } else {
            SPDLOG_ERROR("GetObject error for file: s3://{}/{}, {} {}", bucket_, key_,
                         get_object_outcome.GetError().GetExceptionName(),
                         get_object_outcome.GetError().GetMessage());
            throw std::runtime_error("GetObject error");
        }
    }

    virtual void Seek(size_t pos) override {
        if (pos > size_) {
            SPDLOG_ERROR(
                "Try to seek position {} on object {}, which is larger than total size: {}", pos,
                key_, size_);
            throw std::runtime_error("Seek error");
        }
        pos_ = pos;
        read_buf_.Clear(); // clear buffer
    }

    virtual size_t Tell(void) override {
        // since buffer might have some room,
        // we need to substract the unconsumed size in buffer.
        return pos_ - read_buf_.LeftRoom();
    }

    virtual void Write(const void *ptr, size_t size) override { write_buf_.Write(ptr, size); }

  private:
    Aws::S3::S3Client client_;
    URI path_;
    Aws::String bucket_;
    Aws::String key_; // filename

    // use for read
    size_t pos_;
    size_t size_;
    ReadBuffer read_buf_;

    // use for write
    WriteBuffer write_buf_;
    bool is_write_;
};

size_t ReadBuffer::FillBuffer(Stream *s) {
    Clear();
    buf_.resize(read_buf_size);
    S3SDKStream *s3s = static_cast<S3SDKStream *>(s);
    size_ = s3s->ActualRead((void *)buf_.data(), read_buf_size);
    return size_;
}

/*!
 * \brief open a stream, will report error and exit if bad thing happens
 * NOTE: the Stream can continue to work even when filesystem was destructed
 * \param path path to file
 * \param uri the uri of the input
 * \param flag can be "w", "r", "a"
 * \param allow_null whether NULL can be returned, or directly report error
 * \return the created stream, can be NULL when allow_null == true and file do not exist
 */
Stream *S3FileSystem::Open(const URI &path, const char *const flag, bool allow_null) {
    S3SDKStream *s = new S3SDKStream;
    const size_t flen = strlen(flag);
    const bool read_only = std::find(flag, flag + flen + 1, 'w') == flag + flen + 1 ? true : false;
    s->Open(path, read_only);
    return s;
}
/*!
 * \brief open a seekable stream for read
 * \param path the path to the file
 * \param allow_null whether NULL can be returned, or directly report error
 * \return the created stream, can be NULL when allow_null == true and file do not exist
 */
SeekStream *S3FileSystem::OpenForRead(const URI &path, bool allow_null) {
    S3SDKStream *s = new S3SDKStream;
    s->Open(path, true);
    return s;
}

S3FileSystem::S3FileSystem() {}

S3FileSystem *S3FileSystem::GetInstance(void) {
    static S3FileSystem instance;
    return &instance;
}

} // namespace metaspore
