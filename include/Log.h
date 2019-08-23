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

#include <stdio.h>

#define VERBOSE 1
#define DEBUG 2
#define INFO 3
#define WARN 4
#define ERROR 5
#define FATAL 6

#define ALOG(level, ...) printf(__VA_ARGS__);printf("\n");

#define ALOGV(...) ALOG(VERBOSE, __VA_ARGS__);
#define ALOGD(...) ALOG(DEBUG, __VA_ARGS__);
#define ALOGI(...) ALOG(INFO, __VA_ARGS__);
#define ALOGW(...) ALOG(WARN, __VA_ARGS__);
#define ALOGE(...) ALOG(ERROR, __VA_ARGS__);
#define ALOGF(...) ALOG(ASSERT, __VA_ARGS__);
