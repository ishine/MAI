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

#include <stdio.h>
#include <stdarg.h>
#include <sys/syscall.h>
#include <unistd.h>

#define MAI_VERBOSE 1
#define MAI_DEBUG 2
#define MAI_INFO 3
#define MAI_WARN 4
#define MAI_ERROR 5
#define MAI_FATAL 6

inline void mai_printf(int level, const char* file,
        const char* func, int lineNum, const char* fmt, ...) {
    if (fmt == NULL) {
        return;
    }
    char buf[256 * 2];

    va_list arg;
    va_start(arg, fmt);
    vsprintf(buf, fmt, arg);
    va_end(arg);
    printf("%s--%s:%d | %s\n", file, func, lineNum, buf);
}

#define ALOG(level, ...) mai_printf(level, __FILE__,  __FUNCTION__, (int)__LINE__, __VA_ARGS__);

#define ALOGV(...) ALOG(MAI_VERBOSE, __VA_ARGS__);
#define ALOGD(...) ALOG(MAI_DEBUG, __VA_ARGS__);
#define ALOGI(...) ALOG(MAI_INFO, __VA_ARGS__);
#define ALOGW(...) ALOG(MAI_WARN, __VA_ARGS__);
#define ALOGE(...) ALOG(MAI_ERROR, __VA_ARGS__);
#define ALOGF(...) ALOG(MAI_FATAL, __VA_ARGS__);
