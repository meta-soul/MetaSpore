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

/*!
 *  Copyright (c) 2015 by Contributors
 * \file s3_sdk_filesys.h
 * \brief S3 access module
 * \author Tianqi Chen
 */
#ifndef DMLC_IO_S3_FILESYS_H_
#define DMLC_IO_S3_FILESYS_H_

#include "metaspore/filesys.h"
#include <string>

namespace metaspore {
/*! \brief AWS S3 filesystem */
class S3FileSystem : public FileSystem {
  public:
    /*! \brief destructor */
    virtual ~S3FileSystem() {}
    /*!
     * \brief get information about a path
     * \param path the path to the file
     * \return the information about the file
     */
    virtual FileInfo GetPathInfo(const URI &path) override;
    /*!
     * \brief list files in a directory
     * \param path to the file
     * \param out_list the output information about the files
     */
    virtual void ListDirectory(const URI &path, std::vector<FileInfo> *out_list) override;
    /*!
     * \brief open a stream, will report error and exit if bad thing happens
     * NOTE: the Stream can continue to work even when filesystem was destructed
     * \param path path to file
     * \param uri the uri of the input
     * \param flag can be "w", "r", "a"
     * \param allow_null whether NULL can be returned, or directly report error
     * \return the created stream, can be NULL when allow_null == true and file do not exist
     */
    virtual Stream *Open(const URI &path, const char *const flag, bool allow_null) override;
    /*!
     * \brief open a seekable stream for read
     * \param path the path to the file
     * \param allow_null whether NULL can be returned, or directly report error
     * \return the created stream, can be NULL when allow_null == true and file do not exist
     */
    virtual SeekStream *OpenForRead(const URI &path, bool allow_null) override;
    /*!
     * \brief get a singleton of S3FileSystem when needed
     * \return a singleton instance
     */
    static S3FileSystem *GetInstance(void);

  private:
    /*! \brief constructor */
    S3FileSystem();
};
} // namespace metaspore
#endif // DMLC_IO_S3_FILESYS_H_
