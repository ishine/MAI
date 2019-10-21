LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE := libprofiling_static

LOCAL_C_INCLUDES := $(LOCAL_PATH)/include            \

LOCAL_SRC_FILES := $(call all-named-files-under,*.cpp,.)

LOCAL_MULTILIB := 64

include $(BUILD_STATIC_LIBRARY)

include $(call all-makefiles-under,$(LOCAL_PATH))
