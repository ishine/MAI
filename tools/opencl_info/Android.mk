LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
LOCAL_C_INCLUDES := $(LOCAL_PATH)/../../source/ops/gpu/runtime \
                    $(LOCAL_PATH)/../../3rd_party/opencl \
                    $(LOCAL_PATH)/../../source \
                    $(LOCAL_PATH)/../.. \

LOCAL_SRC_FILES := Main.cpp
LOCAL_MODULE := opencl_info
LOCAL_MULTILIB := 64
LOCAL_VENDOR_MODULE := true
LOCAL_SHARED_LIBRARIES := libmai

include $(BUILD_EXECUTABLE)
