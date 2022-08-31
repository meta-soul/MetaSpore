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

#include <boost/dll/runtime_symbol_info.hpp>
#include <boost/process/system.hpp>
#include <boost/process/search_path.hpp>
#include <common/test_utils.h>
#include <serving/py_preprocessing_process.h>

using namespace metaspore::serving;

TEST(PyPreprocessingProcessTestSuite, LaunchProcessTest) {
    auto prog_dir = boost::dll::program_location().parent_path();
    auto conf_dir = prog_dir / "testing_preprocessor_conf";
    PyPreprocessingProcess proc;
    proc.set_python_executable(boost::process::search_path("python").string());
    proc.set_virtual_env_dir((prog_dir / "testing_preprocessor_venv").string());
    proc.set_requirement_file((conf_dir / "requirements.txt").string());
    proc.set_service_script_file((prog_dir / "preprocessor_service.py").string());
    proc.set_preprocessor_config_dir(conf_dir.string());
    proc.set_preprocessor_listen_addr("unix://" + (conf_dir / "listen_addr.sock").string());
    absl::Status status = proc.launch();
    int rc = boost::process::system(prog_dir / "testing_preprocessor_venv" / "bin" / "python",
                                    conf_dir / "test_example_preprocessor.py");
    ASSERT_EQ(rc, 0);
    ASSERT_TRUE(status.ok());
}

int main(int argc, char **argv) { return run_all_tests(argc, argv); }
