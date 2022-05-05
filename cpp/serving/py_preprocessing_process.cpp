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
#include <serving/py_preprocessing_process.h>

namespace metaspore::serving {

status PyPreprocessingProcess::launch() {
    namespace bp = boost::process;
    namespace fs = std::filesystem;
    int rc = bp::system(python_executable_, "-m", "venv", "--clear", virtual_env_dir_);
    if (rc != 0)
        return absl::FailedPreconditionError("fail to create virtual env \"" + virtual_env_dir_ + "\"");
    fs::path venv_py = fs::path{virtual_env_dir_} / "bin" / "python";
    rc = bp::system(venv_py.string(), "-m", "pip", "install", "-r", requirement_file_);
    if (rc != 0)
        return absl::FailedPreconditionError("fail to install requirement file \"" + requirement_file_ + "\"");
    bp::child child{venv_py.string(), python_script_file_};
    child.wait();
    if (child.exit_code() != 0)
        return absl::FailedPreconditionError("fail to launch python script \"" + python_script_file_ + "\"");
    return absl::OkStatus();
}

} // namespace metaspore::serving
