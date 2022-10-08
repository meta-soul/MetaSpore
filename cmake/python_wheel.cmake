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
    python/metaspore/algos/__init__.py
    python/metaspore/algos/autoint_net.py
    python/metaspore/algos/dcn_net.py
    python/metaspore/algos/dcn_v2_net.py
    python/metaspore/algos/deepfm_net.py
    python/metaspore/algos/feature/__init__.py
    python/metaspore/algos/feature/neg_sampler.py
    python/metaspore/algos/feature/sequential_encoder.py
    python/metaspore/algos/feature/target_encoder.py
    python/metaspore/algos/feature/woe_encoder.py
    python/metaspore/algos/ffm_net.py
    python/metaspore/algos/fwfm_net.py
    python/metaspore/algos/item_cf_retrieval.py
    python/metaspore/algos/layers.py
    python/metaspore/algos/multitask/__init__.py
    python/metaspore/algos/multitask/esmm/esmm_agent.py
    python/metaspore/algos/multitask/esmm/esmm_net.py
    python/metaspore/algos/multitask/mmoe/mmoe_agent.py
    python/metaspore/algos/multitask/mmoe/mmoe_net.py
    python/metaspore/algos/pipeline/__init__.py
    python/metaspore/algos/pipeline/common_validators.py
    python/metaspore/algos/pipeline/data_loader.py
    python/metaspore/algos/pipeline/deep_ctr.py
    python/metaspore/algos/pipeline/i2i_retrieval.py
    python/metaspore/algos/pipeline/init_spark.py
    python/metaspore/algos/pipeline/mongodb_dumper.py
    python/metaspore/algos/pipeline/popular_retrieval.py
    python/metaspore/algos/pipeline/utils/__init__.py
    python/metaspore/algos/pipeline/utils/class_utils.py
    python/metaspore/algos/pipeline/utils/constants.py
    python/metaspore/algos/pipeline/utils/dict_utils.py
    python/metaspore/algos/pipeline/utils/logger.py
    python/metaspore/algos/pnn_net.py
    python/metaspore/algos/sequential/__init__.py
    python/metaspore/algos/sequential/bst/bst_net.py
    python/metaspore/algos/sequential/dien/dien_agent.py
    python/metaspore/algos/sequential/dien/dien_net.py
    python/metaspore/algos/sequential/din/din_net.py
    python/metaspore/algos/sequential/gru4rec/gru4rec_agent.py
    python/metaspore/algos/sequential/gru4rec/gru4rec_net.py
    python/metaspore/algos/sequential/hrm/hrm_net.py
    python/metaspore/algos/tuner/base_tuner.py
    python/metaspore/algos/twotower/dssm/__init__.py
    python/metaspore/algos/twotower/dssm/dssm_agent.py
    python/metaspore/algos/twotower/dssm/dssm_net.py
    python/metaspore/algos/twotower/__init__.py
    python/metaspore/algos/twotower/simplex/__init__.py
    python/metaspore/algos/twotower/simplex/simplex_agent.py
    python/metaspore/algos/twotower/simplex/simplex_net.py
    python/metaspore/algos/widedeep_net.py
    python/metaspore/algos/xdeepfm_net.py
    python/ps/__init__.py
    python/ps/job.py
)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${wheel_file_name}
                   COMMAND env _METASPORE_SO=${PROJECT_BINARY_DIR}/_metaspore.so
                           ${Python_EXECUTABLE} -m pip wheel ${PROJECT_SOURCE_DIR}
                   MAIN_DEPENDENCY setup.py
                   DEPENDS metaspore_shared ${python_files} install_wheel)
add_custom_target(python_wheel ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${wheel_file_name})
