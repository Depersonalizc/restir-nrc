# Looks for environment variable:
# TCNN_PATH 

# Sets the variables:
# TCNN_INCLUDE_DIRS
# TCNN_LIBRARIES
# TCNN_FOUND

set(TCNN_PATH $ENV{TCNN_PATH})

# If there was no environment variable override for the TCNN_PATH
# try finding it inside the local 3rdparty path.
if ("${TCNN_PATH}" STREQUAL "")
  set(TCNN_PATH "${LOCAL_3RDPARTY}/tiny-cuda-nn")
  set(TCNN_LIBRARIES "${LOCAL_3RDPARTY}/tiny-cuda-nn/build/RelWithDebInfo/tiny-cuda-nn.lib")
  set(TCNN_DEPENDENCIES "${LOCAL_3RDPARTY}/tiny-cuda-nn/dependencies")
endif()

message("TCNN_PATH = " "${TCNN_PATH}")
message("TCNN_LIBRARIES = " "${TCNN_LIBRARIES}")
message("TCNN_DEPENDENCIES = " "${TCNN_DEPENDENCIES}")

find_path( TCNN_INCLUDE_DIRS "tiny-cuda-nn/network.h"
  PATHS /usr/include ${TCNN_PATH}/include )

message("TCNN_INCLUDE_DIRS = " "${TCNN_INCLUDE_DIRS}")

# There are no link libraries inside the (pre-built) MDL SDK. DLLs are loaded manually.
#if (WIN32)
#  set(MDL_SDK_LIBRARY_DIR ${MDL_SDK_PATH}/nt-x86-x64/lib)
#else()
#  set(MDL_SDK_LIBRARY_DIR ${MDL_SDK_PATH}/lib)
#endif()

# message("MDL_SDK_LIBRARY_DIR = " "${MDL_SDK_LIBRARY_DIR}")

#find_library(MDL_SDK_LIBRARIES
#  NAMES MDL_SDK libmdl_sdk
#  PATHS ${MDL_SDK_LIBRARY_DIR} )

#message("MDL_SDK_LIBRARIES = " "${MDL_SDK_LIBRARIES}")

include(FindPackageHandleStandardArgs)

#find_package_handle_standard_args(MDL_SDK DEFAULT_MSG MDL_SDK_INCLUDE_DIRS MDL_SDK_LIBRARIES)
find_package_handle_standard_args(TCNN DEFAULT_MSG TCNN_INCLUDE_DIRS)

#mark_as_advanced(MDL_SDK_INCLUDE_DIRS MDL_SDK_LIBRARIES)
mark_as_advanced(TCNN_INCLUDE_DIRS)

message("TCNN_FOUND = " "${TCNN_FOUND}")
