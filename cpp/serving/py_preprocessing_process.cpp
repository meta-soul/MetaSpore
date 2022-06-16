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

#include <filesystem>
#include <boost/process/system.hpp>
#include <boost/process/pipe.hpp>
#include <boost/process/io.hpp>
#include <metaspore/string_utils.h>
#include <serving/py_preprocessing_process.h>

namespace metaspore::serving {

status PyPreprocessingProcess::launch() {
    namespace bp = boost::process;
    namespace fs = std::filesystem;
    int rc = bp::system(python_executable_, "-m", "venv", virtual_env_dir_);
    if (rc != 0)
        return absl::FailedPreconditionError("fail to create virtual env \"" + virtual_env_dir_ + "\"");
    fs::path venv_py = fs::path{virtual_env_dir_} / "bin" / "python";
    rc = bp::system(venv_py.string(), "-m", "pip", "install", "--upgrade", "pip");
    if (rc != 0)
        return absl::FailedPreconditionError("fail to upgrade pip");
    if (!requirement_file_.empty()) {
        rc = bp::system(venv_py.string(), "-m", "pip", "install", "-r", requirement_file_);
        if (rc != 0)
            return absl::FailedPreconditionError("fail to install requirement file \"" + requirement_file_ + "\"");
    }
    bp::ipstream pipe_stream;
    bp::child child{venv_py.string(), service_script_file_,
                    "--config-dir", preprocessor_config_dir_,
                    "--listen-addr", preprocessor_listen_addr_,
                    bp::std_out > pipe_stream
                   };
    std::string line;
    while (pipe_stream && std::getline(pipe_stream, line) && !line.empty()) {
        if (line.starts_with("input_names=")) {
            input_names_ = parse_name_list(line);
        } else if (line.starts_with("output_names=")) {
            output_names_ = parse_name_list(line);
        }
        if (!input_names_.empty() && !output_names_.empty())
            break;
    }
    if (input_names_.empty() || output_names_.empty()) {
        return absl::FailedPreconditionError("fail to parse input and output names; conf_dir = \"" +
                                             preprocessor_config_dir_ + "\"");
    }
    child_process_ = std::move(child);
    return absl::OkStatus();
}

std::vector<std::string> PyPreprocessingProcess::parse_name_list(const std::string &line) {
    std::string::size_type i = line.find('=') + 1;
    std::string_view str{line.data() + i, line.size() - i};
    auto names = metaspore::SplitStringView(str, ",");
    return {names.begin(), names.end()};
}

} // namespace metaspore::serving
