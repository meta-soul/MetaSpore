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

#include <serving/types.h>
#include <boost/process/child.hpp>

namespace metaspore::serving {

class PyPreprocessingProcess {
public:
    const std::string &get_python_executable() const { return python_executable_; }
    void set_python_executable(std::string value) { python_executable_ = std::move(value); }

    const std::string &get_virtual_env_dir() const { return virtual_env_dir_; }
    void set_virtual_env_dir(std::string value) { virtual_env_dir_ = std::move(value); }

    const std::string &get_requirement_file() const { return requirement_file_; }
    void set_requirement_file(std::string value) { requirement_file_ = std::move(value); }

    const std::string &get_python_script_file() const { return python_script_file_; }
    void set_python_script_file(std::string value) { python_script_file_ = std::move(value); }

    status launch();

private:
    std::string python_executable_;
    std::string virtual_env_dir_;
    std::string requirement_file_;
    std::string python_script_file_;
    boost::process::child child_process_;
};

} // namespace metaspore::serving
