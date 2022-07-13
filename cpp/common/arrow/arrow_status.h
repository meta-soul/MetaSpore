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

#include <absl/status/status.h>
#include <arrow/status.h>

namespace metaspore::serving {

class ArrowStatusToAbsl {
  public:
    static inline absl::Status arrow_status_to_absl(const arrow::Status &as) {
        using arrow::StatusCode;
        switch (as.code()) {
            [[likely]] case StatusCode::OK : return absl::OkStatus();
        case StatusCode::OutOfMemory:
            return absl::ResourceExhaustedError(as.ToString());
        case StatusCode::KeyError:
            return absl::NotFoundError(as.ToString());
        case StatusCode::TypeError:
            return absl::InvalidArgumentError(as.ToString());
        case StatusCode::CapacityError:
            return absl::FailedPreconditionError(as.ToString());
        case StatusCode::IndexError:
            return absl::OutOfRangeError(as.ToString());
        case StatusCode::Cancelled:
            return absl::CancelledError(as.ToString());
        case StatusCode::UnknownError:
            return absl::UnknownError(as.ToString());
        case StatusCode::NotImplemented:
            return absl::UnimplementedError(as.ToString());
        case StatusCode::AlreadyExists:
            return absl::AlreadyExistsError(as.ToString());
        default:
            return absl::InternalError(as.ToString());
        }
    }
};

} // namespace metaspore::serving