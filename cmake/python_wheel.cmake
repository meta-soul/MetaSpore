#
# Copyright 2022 DMetaSoul
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/install_wheel.stamp
    COMMAND ${Python_EXECUTABLE} -m pip install wheel
    COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/install_wheel.stamp
    DEPENDS Python::Interpreter)
add_custom_target(install_wheel DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/install_wheel.stamp)

set(python_files
    pyproject.toml
    setup.py
    python/metaspore/__init__.py
    python/metaspore/initializer.py
    python/metaspore/updater.py
    python/metaspore/model.py
    python/metaspore/distributed_trainer.py
    python/metaspore/distributed_tensor.py
    python/metaspore/agent.py
    python/metaspore/metric.py
    python/metaspore/loss_utils.py
    python/metaspore/feature_group.py
    python/metaspore/embedding.py
    python/metaspore/cast.py
    python/metaspore/input.py
    python/metaspore/output.py
    python/metaspore/url_utils.py
    python/metaspore/s3_utils.py
    python/metaspore/file_utils.py
    python/metaspore/name_utils.py
    python/metaspore/network_utils.py
    python/metaspore/shell_utils.py
    python/metaspore/stack_trace_utils.py
    python/metaspore/ps_launcher.py
    python/metaspore/job_utils.py
    python/metaspore/schema_utils.py
    python/metaspore/estimator.py
    python/metaspore/two_tower_ranking.py
    python/metaspore/two_tower_retrieval.py
    python/metaspore/swing_retrieval.py
    python/metaspore/experiment.py
    python/metaspore/spark.py
    python/metaspore/patching_pickle.py
    python/metaspore/nn/__init__.py
    python/metaspore/nn/normalization.py
    python/metaspore/nn/fm.py
    python/metaspore/nn/wide_and_deep.py
    python/metaspore/nn/deep_fm.py
    python/metaspore/compat/__init__.py
    python/metaspore/compat/ps/__init__.py
    python/ps/__init__.py
    python/ps/job.py
)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${wheel_file_name}
                   COMMAND env _METASPORE_SO=${PROJECT_BINARY_DIR}/_metaspore.so
                           ${Python_EXECUTABLE} -m pip wheel ${PROJECT_SOURCE_DIR}
                   MAIN_DEPENDENCY setup.py
                   DEPENDS metaspore_shared ${python_files} install_wheel)
add_custom_target(python_wheel ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${wheel_file_name})
