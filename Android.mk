LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE := libmai

LOCAL_MODULE_CLASS := SHARED_LIBRARIES

LOCAL_C_INCLUDES := $(LOCAL_PATH)/include            \
                    $(LOCAL_PATH)/source/            \
                    $(LOCAL_PATH)/source/core        \
                    $(LOCAL_PATH)/source/ops/cpu/ref \
                    $(LOCAL_PATH)/source/ops/cpu/ \
                    $(LOCAL_PATH)/source/ops/gpu/ \
                    $(LOCAL_PATH)/3rd_party/openmp/include \
                    $(LOCAL_PATH)/3rd_party/opencl \
                    $(call intermediates-dir-for, SHARED_LIBRARIES, $(LOCAL_MODULE))/proto

LOCAL_SRC_FILES := $(call all-cpp-files-under,source/core source/ops/cpu source/util source/ops/gpu)

LOCAL_MULTILIB := 64
LOCAL_STATIC_LIBRARIES := libomp_static libprofiling_static

LOCAL_CFLAGS += -std=c++11 -fopenmp -O3
LOCAL_CFLAGS += -DMAI_NEON_ENABLED

include $(BUILD_SHARED_LIBRARY)

include $(call all-makefiles-under,$(LOCAL_PATH))
