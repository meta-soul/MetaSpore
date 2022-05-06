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
#include <serving/test_utils.h>
#include <serving/py_preprocessing_process.h>

namespace fs = std::filesystem;
using namespace metaspore::serving;

TEST(PyPreprocessingProcessTestSuite, LaunchTest) {
    PyPreprocessingProcess proc;
    proc.set_python_executable("/usr/bin/python");
    proc.set_virtual_env_dir((fs::current_path() / "preprocessor_venv").string());
    proc.set_requirement_file((fs::current_path() / "requirements.txt").string());
    proc.set_python_script_file((fs::current_path() / "preprocessor_service.py").string());
    absl::Status status = proc.launch();
    ASSERT_TRUE(status.ok());
}

int main(int argc, char **argv) { return run_all_tests(argc, argv); }
