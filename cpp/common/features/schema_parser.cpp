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

#include <arrow/compute/api.h>
#include <arrow/compute/exec/exec_plan.h>
#include <boost/algorithm/string.hpp>
#include <boost/spirit/home/x3.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/transform.hpp>

#include <common/logger.h>
#include <common/features/feature_compute_exec.h>
#include <common/features/feature_compute_funcs.h>
#include <common/features/schema_parser.h>

#include <filesystem>

namespace metaspore {

using boost::spirit::x3::_attr;
using boost::spirit::x3::char_;
using namespace std::placeholders;
namespace fs = std::filesystem;
namespace cp = arrow::compute;

// identifier rule for table name or column name
auto ident = +(char_("a-zA-z_0-9"));

status FeatureSchemaParser::parse(const std::string &file, FeatureComputeExec &exec) {
    std::ifstream ifs(file);
    if (!ifs) {
        return absl::InternalError(fmt::format("Open file {} failed {}", file, strerror(errno)));
    }
    auto status = parse_new_format(ifs, exec);
    if (!status.ok()) {
        CALL_AND_RETURN_IF_STATUS_NOT_OK(parse_table_name_from_path(file, exec));
        std::ifstream ifs(file); // we need to open this file again
        int feature_count = 0;
        CALL_AND_RETURN_IF_STATUS_NOT_OK(parse_hash_and_combine(ifs, exec, feature_count));
    }
    return absl::OkStatus();
}

status FeatureSchemaParser::parse_table_name_from_path(const std::string &file,
                                                       FeatureComputeExec &exec) {
    auto dir_name = fs::path(file).parent_path().filename().string();
    spdlog::info("New schema not found for {}, parsing dir name {} as table name, and all combines",
                 file, dir_name);
    auto name = dir_name;
    boost::trim_if(name, boost::is_any_of("/\\"));
    if (name.empty()) {
        return absl::InvalidArgumentError(fmt::format("{} cannot be a valid table name", dir_name));
    }
    if (boost::starts_with(name, "sparse_")) {
        // a convertion with offline model export, which is of name sparse_'layer_name',
        // so we remove the starting 'sparse_' and use left as table name
        auto subname = name.substr(7);
        if (subname.empty()) {
            return absl::InvalidArgumentError(fmt::format(
                "{} should contain a table name after sparse_ to be a valid table name", dir_name));
        }
        name = subname;
    }
    return exec.add_source(name);
}

status FeatureSchemaParser::parse_new_format(std::istream &is, FeatureComputeExec &exec) {
    std::string line;
    int feature_count = 0;
    while (std::getline(is, line)) {
        boost::trim_if(line, boost::is_any_of(" \t\n\r"));
        if (line.empty())
            continue;

        if (!boost::starts_with(line, "#")) {
            continue;
        }

        boost::trim_if(line, boost::is_any_of("# "));
        if (line.empty())
            continue;

        if (line == "join") {
            CALL_AND_RETURN_IF_STATUS_NOT_OK(parse_table_join(is, exec));
        } else if (line == "combine") {
            CALL_AND_RETURN_IF_STATUS_NOT_OK(parse_hash_and_combine(is, exec, feature_count));
        } else if (boost::starts_with(line, "table:")) {
            auto table_name = parse_table_name_from_config(line);
            CALL_AND_RETURN_IF_STATUS_NOT_OK(exec.add_source(table_name));
        } else {
            continue;
        }
    }
    if (exec.get_input_names().empty()) {
        return absl::NotFoundError("The schema file is not in new format");
    }
    return absl::OkStatus();
}

std::string FeatureSchemaParser::parse_table_name_from_config(const std::string &line) {
    std::string table_name;
    auto fn = [&](auto &context) { table_name = _attr(context); };
    auto rule = "table: " >> ident[fn];
    if (boost::spirit::x3::parse(line.begin(), line.end(), rule)) {
        return table_name;
    } else {
        return "";
    }
}

status FeatureSchemaParser::parse_table_join(std::istream &is, FeatureComputeExec &exec) {
    // parse join directives in form left_table_name#right_table_name(join_key)=>output_name
    std::string line;
    std::string left_table;
    std::string right_table;
    std::string join_key;
    std::string output_name;

    auto assign = [](auto &context, std::string &name) { name = _attr(context); };

#define BIND(var) std::bind(assign, _1, std::ref(var))

    auto join_rule = ident[BIND(left_table)] >> '#' >> ident[BIND(right_table)] >> '(' >>
                     ident[BIND(join_key)] >> ")=>" >> ident[BIND(output_name)];
    while (std::getline(is, line)) {
        boost::trim_if(line, boost::is_any_of("# \t\n\r"));
        if (line.empty())
            break;

        if (boost::spirit::x3::parse(line.begin(), line.end(), join_rule)) {
            spdlog::info("Parsed join: {}#{}({})=>{}\n", left_table, right_table, join_key,
                         output_name);
            CALL_AND_RETURN_IF_STATUS_NOT_OK(exec.add_join_plan(
                left_table, right_table, cp::JoinType::LEFT_OUTER, {join_key}, {join_key}));
        } else {
            auto m = fmt::format("Parsing join rule failed {}", line);
            spdlog::error(m);
            return absl::InvalidArgumentError(m);
        }
    }
    return absl::OkStatus();
}

status FeatureSchemaParser::parse_hash_and_combine(std::istream &is, FeatureComputeExec &exec, int &feature_count) {
    // parse combine directives in form column1#column2#columnN
    std::string line;
    // combine rule is # seperated identifier list
    std::vector<std::string> feature_columns;
    auto push = [&](auto &context) { feature_columns.push_back(_attr(context)); };
    auto combine_rule = ident[push] % '#';
    std::vector<cp::Expression> expressions;
    while (std::getline(is, line)) {
        fmt::print("{}\n", line);
        feature_columns.clear();
        boost::trim_if(line, boost::is_any_of("# \t\n\r"));
        if (line.empty())
            break;

        if (boost::spirit::x3::parse(line.begin(), line.end(), combine_rule)) {
            if (feature_columns.empty())
                continue;
            if (feature_columns.size() == 1) {
                // only one column, just create a bkdr_hash expression
                expressions.push_back(
                    cp::call("bkdr_hash", {cp::field_ref(feature_columns[0])},
                             StringBKDRHashFunctionOption::Make(feature_columns[0])));
                spdlog::info("add expr {}", expressions.back().ToString());
            } else {
                // more than one column, create bkdr_hash for each one and bkdr_hash_combine them
                // all
                auto subexpressions = feature_columns |
                                      ranges::views::transform([](const std::string &name) {
                                          return cp::call("bkdr_hash", {cp::field_ref(name)},
                                                          StringBKDRHashFunctionOption::Make(name));
                                      }) |
                                      ranges::to<std::vector>();
                expressions.push_back(cp::call("bkdr_hash_combine", std::move(subexpressions),
                                               BKDRHashCombineFunctionOption::Make()));
                spdlog::info("add expr {}", expressions.back().ToString());
            }
            feature_count++;
        } else {
            auto m = fmt::format("Parsing combine rule failed {}", line);
            spdlog::error(m);
            return absl::InvalidArgumentError(m);
        }
    }
    return exec.add_projection(std::move(expressions));
}
} // namespace metaspore
