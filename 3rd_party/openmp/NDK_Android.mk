LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
LOCAL_MODULE := omp_static
LOCAL_SRC_FILES := lib/libomp.a
include $(PREBUILT_STATIC_LIBRARY)
