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

#pragma once

#include <memory>
#include <stdint.h>
#include <string>

namespace metaspore {

class Stream { // NOLINT(*)
  public:
    /*!
     * \brief reads data from a stream
     * \param ptr pointer to a memory buffer
     * \param size block size
     * \return the size of data read
     */
    virtual size_t Read(void *ptr, size_t size) = 0;
    /*!
     * \brief writes data to a stream
     * \param ptr pointer to a memory buffer
     * \param size block size
     */
    virtual void Write(const void *ptr, size_t size) = 0;
    /*! \brief virtual destructor */
    virtual ~Stream(void) {}
    /*!
     * \brief generic factory function
     *  create an stream, the stream will close the underlying files upon deletion
     *
     * \param uri the uri of the input currently we support
     *            hdfs://, s3://, and file:// by default file:// will be used
     * \param flag can be "w", "r", "a"
     * \param allow_null whether NULL can be returned, or directly report error
     * \return the created stream, can be NULL when allow_null == true and file do not exist
     */
    static Stream *Create(const char *uri, const char *const flag, bool allow_null = false);
};

/*! \brief interface of i/o stream that support seek */
class SeekStream : public Stream {
  public:
    // virtual destructor
    virtual ~SeekStream(void) {}
    /*! \brief seek to certain position of the file */
    virtual void Seek(size_t pos) = 0;
    /*! \brief tell the position of the stream */
    virtual size_t Tell(void) = 0;
    /*!
     * \brief generic factory function
     *  create an SeekStream for read only,
     *  the stream will close the underlying files upon deletion
     *  error will be reported and the system will exit when create failed
     * \param uri the uri of the input currently we support
     *            hdfs://, s3://, and file:// by default file:// will be used
     * \param allow_null whether NULL can be returned, or directly report error
     * \return the created stream, can be NULL when allow_null == true and file do not exist
     */
    static SeekStream *CreateForRead(const char *uri, bool allow_null = false);
};

class InputStream {
  public:
    explicit InputStream(const std::string &url);
    size_t Read(void *buffer, size_t size);

  private:
    std::unique_ptr<Stream> stream_;
};

class OutputStream {
  public:
    explicit OutputStream(const std::string &url);
    void Write(const void *buffer, size_t size);

  private:
    std::unique_ptr<Stream> stream_;
};

void StreamWriteAll(const std::string &url, const char *data, size_t size);
void StreamWriteAll(const std::string &url, const std::string &data);
void StreamReadAll(const std::string &url, char *data, size_t size);
std::string StreamReadAll(const std::string &url);

void MakeLocalDirectories(const std::string &path, mode_t mode);
void EnsureLocalDirectory(const std::string &dir_path);
std::string DirName(const std::string &path);
std::string JoinPath(const std::string &dir_path, const std::string &file_name);

} // namespace metaspore
