LOCAL_PATH := $(call my-dir)
MY_PATH := $(LOCAL_PATH)/../../

include $(LOCAL_PATH)/mai_definitions.mk

#gtest
include $(MY_PATH)/3rd_party/gtest/Android.mk

#openmp
include $(MY_PATH)/3rd_party/openmp/NDK_Android.mk

#profiling & benchmark & opencl_info
include $(call all-makefiles-under,$(MY_PATH)/tools/)

#libmai
include $(MY_PATH)/Android.mk

#operator_test
include $(MY_PATH)/test/optest/Android.mk

#optimizer_test
include $(MY_PATH)/test/optimizertest/Android.mk
