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

#include <gflags/gflags.h>

namespace metaspore {

DEFINE_int32(log_level, 2, "Global log level");
DEFINE_uint64(compute_thread_num, 4UL, "Thread number for computing");
DEFINE_uint64(background_thread_num, 2UL, "Thread number for background tasks");

} // namespace metaspore
