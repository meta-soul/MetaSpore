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

#include <common/types.h>
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

    const std::string &get_service_script_file() const { return service_script_file_; }
    void set_service_script_file(std::string value) { service_script_file_ = std::move(value); }

    const std::string &get_preprocessor_config_dir() const { return preprocessor_config_dir_; }
    void set_preprocessor_config_dir(std::string value) { preprocessor_config_dir_ = std::move(value); }

    const std::string &get_preprocessor_listen_addr() const { return preprocessor_listen_addr_; }
    void set_preprocessor_listen_addr(std::string value) { preprocessor_listen_addr_ = std::move(value); }

    const std::vector<std::string> &get_input_names() const { return input_names_; }
    const std::vector<std::string> &get_output_names() const { return output_names_; }

    status launch();

private:
    static std::vector<std::string> parse_name_list(const std::string &line);

    std::string python_executable_;
    std::string virtual_env_dir_;
    std::string requirement_file_;
    std::string service_script_file_;
    std::string preprocessor_config_dir_;
    std::string preprocessor_listen_addr_;
    boost::process::child child_process_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
};

} // namespace metaspore::serving
