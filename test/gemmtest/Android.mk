LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
#LOCAL_ARM_MODE := arm
#LOCAL_ARM_NEON := true
LOCAL_C_INCLUDES := $(LOCAL_PATH)/core \
                    $(LOCAL_PATH)/../../3rd_party/gtest/include

LOCAL_SRC_FILES := $(call all-cpp-files-under,core)
LOCAL_MODULE := gemm_test
LOCAL_CFLAGS := -std=c++11 -fopenmp
LOCAL_MULTILIB := 64
LOCAL_STATIC_LIBRARIES += libgtest_static libomp_static

include $(BUILD_EXECUTABLE)

include $(LOCAL_PATH)/../../3rd_party/openmp/NDK_Android.mk
