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

#include <istream>
#include <memory>

#include <arrow/compute/exec/exec_plan.h>
#include <arrow/compute/exec/options.h>
#include <arrow/record_batch.h>

#include <common/types.h>

namespace metaspore {

class FeatureComputeContext;
class FeatureComputeExecContext;

class FeatureComputeExec {
  public:
    FeatureComputeExec();
    ~FeatureComputeExec();
    FeatureComputeExec(FeatureComputeExec &&);

    status add_source(const std::string &name);

    status add_join_plan(const std::string &left_source_name, const std::string &right_source_name,
                         arrow::compute::JoinType join_type,
                         const std::vector<std::string> &left_key_names,
                         const std::vector<std::string> &right_key_names);

    status add_projection(std::vector<arrow::compute::Expression> expressions);

    result<std::shared_ptr<FeatureComputeExecContext>> start_plan();

    status set_input_schema(std::shared_ptr<FeatureComputeExecContext> &ctx,
                            const std::string &source_name, std::shared_ptr<arrow::Schema> schema);

    status build_plan(std::shared_ptr<FeatureComputeExecContext> &ctx);

    status feed_input(std::shared_ptr<FeatureComputeExecContext> &ctx,
                      const std::string &source_name, std::shared_ptr<arrow::RecordBatch> batch);

    awaitable_result<std::shared_ptr<arrow::RecordBatch>>
    execute(std::shared_ptr<FeatureComputeExecContext> &ctx);

    status finish_plan(std::shared_ptr<FeatureComputeExecContext> &ctx);

    std::vector<std::string> get_input_names() const;

  protected:
    void finish_join(arrow::compute::ExecNode *node);

  private:
    std::unique_ptr<FeatureComputeContext> context_;
};

} // namespace metaspore
