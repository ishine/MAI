LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_ARM_MODE := arm
LOCAL_ARM_NEON := true
LOCAL_C_INCLUDES := $(LOCAL_PATH)/core \
                    $(LOCAL_PATH)/../../3rd_party/gtest/include \
                    $(LOCAL_PATH)/func

LOCAL_SRC_FILES := $(call all-cpp-files-under,core units func)

#LOCAL_SRC_FILES += func/Transpose.s

LOCAL_MODULE := neon_test
LOCAL_MULTILIB := 64
LOCAL_VENDOR_MODULE := true

LOCAL_STATIC_LIBRARIES := libgtest_static

include $(BUILD_EXECUTABLE)
