LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_C_INCLUDES := $(LOCAL_PATH)/core \
                    $(LOCAL_PATH)/../../3rd_party/gtest/include \
                    $(LOCAL_PATH)/../.. \
                    $(LOCAL_PATH)/../../source \
                    $(LOCAL_PATH)/../../include \

LOCAL_SRC_FILES := $(call all-cpp-files-under,core units)

LOCAL_MODULE := optimizer_test
LOCAL_MULTILIB := 64
LOCAL_VENDOR_MODULE := true

LOCAL_STATIC_LIBRARIES := libgtest_static libprofiling_static
LOCAL_SHARED_LIBRARIES := libmai

include $(BUILD_EXECUTABLE)
