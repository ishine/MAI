// Copyright 2019 MAI. All rights reserved.
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

#pragma once

#include "include/Log.h"

#define unlikely(x) __builtin_expect(!!(x), 0)
#define likely(x) __builtin_expect(!!(x), 1)

#define MAI_CHECK(condition, ...)      \
    do {                               \
        if (unlikely(!(condition))) {  \
            ALOGE(__VA_ARGS__);        \
            abort();                   \
        }                              \
    } while(0)                         \

#define MAI_CHECK_NULL(ptr)            \
    do {                               \
        if (unlikely(NULL == ptr)) {   \
            ALOGE(#ptr " cannot be null");   \
            abort();                   \
        }                              \
    } while(0)                         \

#define MAI_UNUSED(para) (void)para

#define MAI_ABORT(...) \
    do { \
        ALOGF(__VA_ARGS__); \
        abort(); \
    } while(0) \

