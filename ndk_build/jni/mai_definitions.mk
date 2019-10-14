###########################################################
## Find all of the cpp files under the named directories.
## LOCAL_CPP_EXTENSION is respected if set.
## Meant to be used like:
##    SRC_FILES := $(call all-cpp-files-under,src tests)
###########################################################

define all-cpp-files-under
$(sort $(patsubst ./%,%, \
    $(shell cd $(LOCAL_PATH) ; \
        find -L $(1) -name "*$(or $(LOCAL_CPP_EXTENSION),.cpp)" -and -not -name ".*") \
))
endef

define all-named-files-under
$(call find-files-in-subdirs,$(LOCAL_PATH),"$(1)",$(2))
endef

###########################################################
# Use utility find to find given files in the given subdirs.
# # This function uses $(1), instead of LOCAL_PATH as the base.
# # $(1): the base dir, relative to the root of the source tree.
# # $(2): the file name pattern to be passed to find as "-name".
# # $(3): a list of subdirs of the base dir.
# # Returns: a list of paths relative to the base dir.
# ###########################################################

define find-files-in-subdirs
$(sort $(patsubst ./%,%, \
  $(shell cd $(1) ; \
           find -L $(3) -name $(2) -and -not -name ".*") \
 ))
endef

