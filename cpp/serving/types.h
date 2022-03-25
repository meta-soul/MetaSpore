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

#include <absl/status/statusor.h>
#include <arrow/result.h>
#include <boost/asio/awaitable.hpp>
#include <boost/asio/co_spawn.hpp>
#include <boost/asio/detached.hpp>

#include <serving/arrow_status.h>

namespace metaspore::serving {

template <typename T> using awaitable = boost::asio::awaitable<T>;

template <typename T> using result = absl::StatusOr<T>;

using status = absl::Status;

#define RETURN_IF_STATUS_NOT_OK(status)                                                            \
    if (!status.ok()) {                                                                            \
        return status;                                                                             \
    }

#define CO_RETURN_IF_STATUS_NOT_OK(status)                                                         \
    if (!status.ok()) {                                                                            \
        co_return status;                                                                          \
    }

struct GetStatus {
    static const status &get(const status &s) { return s; }

    static status get(arrow::Status s) { return ArrowStatusToAbsl::arrow_status_to_absl(s); }

    template <typename T> static const status &get(const result<T> &r) { return r.status(); }

    template <typename T> static status get(const arrow::Result<T> &r) {
        return ArrowStatusToAbsl::arrow_status_to_absl(r.status());
    }
};

#define CALL_AND_RETURN_IF_STATUS_NOT_OK(expr)                                                     \
    do {                                                                                           \
        auto s = GetStatus::get(expr);                                                             \
        RETURN_IF_STATUS_NOT_OK(s);                                                                \
    } while (false)

#define CALL_AND_CO_RETURN_IF_STATUS_NOT_OK(expr)                                                  \
    do {                                                                                           \
        auto s = GetStatus::get(expr);                                                             \
        CO_RETURN_IF_STATUS_NOT_OK(s);                                                             \
    } while (false)

#define CO_AWAIT_AND_CO_RETURN_IF_STATUS_NOT_OK(expr)                                              \
    do {                                                                                           \
        auto s = GetStatus::get(co_await expr);                                                    \
        CO_RETURN_IF_STATUS_NOT_OK(s);                                                             \
    } while (false)

#define __ASSIGN_RESULT_OR_RETURN_NOT_OK(result_name, lhs, rexpr)                                  \
    auto &&result_name = (rexpr);                                                                  \
    CALL_AND_RETURN_IF_STATUS_NOT_OK(result_name);                                                 \
    lhs = std::move(*result_name);

#define __MS_CONCAT(x, y) x##y

#define RESULT_NAME(x, y) __MS_CONCAT(x, y)

#define ASSIGN_RESULT_OR_RETURN_NOT_OK(lhs, rexpr)                                                 \
    __ASSIGN_RESULT_OR_RETURN_NOT_OK(RESULT_NAME(__r__, __COUNTER__), lhs, rexpr);

#define __ASSIGN_RESULT_OR_CO_RETURN_NOT_OK(result_name, lhs, rexpr)                               \
    auto &&result_name = (rexpr);                                                                  \
    CALL_AND_CO_RETURN_IF_STATUS_NOT_OK(result_name);                                              \
    lhs = std::move(*result_name);

#define ASSIGN_RESULT_OR_CO_RETURN_NOT_OK(lhs, rexpr)                                              \
    __ASSIGN_RESULT_OR_CO_RETURN_NOT_OK(RESULT_NAME(__r__, __COUNTER__), lhs, rexpr);

#define __CO_ASSIGN_RESULT_OR_CO_RETURN_NOT_OK(result_name, lhs, rexpr)                            \
    auto &&result_name = (co_await rexpr);                                                         \
    CALL_AND_CO_RETURN_IF_STATUS_NOT_OK(result_name);                                              \
    lhs = std::move(*result_name);

#define CO_ASSIGN_RESULT_OR_CO_RETURN_NOT_OK(lhs, rexpr)                                           \
    __CO_ASSIGN_RESULT_OR_CO_RETURN_NOT_OK(RESULT_NAME(__r__, __COUNTER__), lhs, rexpr);

template <typename T> using awaitable_result = awaitable<result<T>>;

using awaitable_status = awaitable<status>;

} // namespace metaspore::serving