LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE := libgtest_static

LOCAL_C_INCLUDES := $(LOCAL_PATH)/include            \

LOCAL_SRC_FILES := $(call all-named-files-under,*.cc,src)

LOCAL_CPP_EXTENSION := .cc

LOCAL_MULTILIB := 64

include $(BUILD_STATIC_LIBRARY)

include $(call all-makefiles-under,$(LOCAL_PATH))
