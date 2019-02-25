##############################################################################
#
# CUDA Data Parallel Primitives (CUDPP) CMake file
#
###############################################################################

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  )


#set (HFILES
#  cuda_wrapper_interface.h
#  )

set (CUHFILES
  cuda_util.cuh
  )

set(CUFILES
  cuda_util.cu
  cuda_wrapper.cu
  )

set(HFILES_PUBLIC
  cuda_wrapper_interface.h
  )

source_group("CUDA Source Files" FILES ${CUFILES})
source_group("CUDA Header Files" FILES ${CUHFILES})

#set(GENCODE_SM12 -gencode=arch=compute_12,code=sm_12 -gencode=arch=compute_12,code=compute_12)
#set(GENCODE_SM13 -gencode=arch=compute_13,code=sm_13 -gencode=arch=compute_13,code=compute_13)
#set(GENCODE_SM20 -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=compute_20)
#set(GENCODE_SM21 -gencode=arch=compute_20,code=sm_21 -gencode=arch=compute_20,code=compute_20)
set(GENCODE_SM30 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_30,code=compute_30)
set(GENCODE_SM35 -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_35,code=compute_35)
set(GENCODE_SM50 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_50,code=compute_50)
set(GENCODE_SM61 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_61,code=compute_61)


#set(GENCODE -gencode=arch=compute_20,code=compute_20) # at least generate PTX

#option(CUDPP_GENCODE_SM12
#       "ON to generate code for Compute Capability 1.2 devices (e.g. GeForce GT 330M)"
#       OFF)

#option(CUDPP_GENCODE_SM13
#       "ON to generate code for Compute Capability 1.3 devices (e.g. Tesla C1060)"
#       OFF)

#option(CUDPP_GENCODE_SM20
#       "ON to generate code for Compute Capability 2.0 devices (e.g. Tesla C2050)"
#       OFF)

#option(CUDPP_GENCODE_SM21
#       "ON to generate code for Compute Capability 2.1 devices (e.g. NVS 4200M)"
#       OFF)

option(CUDPP_GENCODE_SM30
       "ON to generate code for Compute Capability 3.0 devices (e.g. Tesla K10)"
       OFF)

option(CUDPP_GENCODE_SM35
       "ON to generate code for Compute Capability 3.5 devices (e.g. Tesla K20)"
       OFF)

option(CUDPP_GENCODE_SM50
       "ON to generate code for Compute Capability 5.0 devices (e.g. GeForce GTX 750)"
       OFF)

option(CUDPP_GENCODE_SM61
       "ON to generate code for Compute Capability 6.1 devices (e.g. GeForce GTX 1060)"
       ON)

#if (CUDPP_GENCODE_SM12)
#  set(GENCODE ${GENCODE} ${GENCODE_SM12})
#endif(CUDPP_GENCODE_SM12)

#if (CUDPP_GENCODE_SM13)
#  set(GENCODE ${GENCODE} ${GENCODE_SM13})
#endif(CUDPP_GENCODE_SM13)

#if (CUDPP_GENCODE_SM20)
#  set(GENCODE ${GENCODE} ${GENCODE_SM20})
#endif(CUDPP_GENCODE_SM20)

#if (CUDPP_GENCODE_SM21)
#  set(GENCODE ${GENCODE} ${GENCODE_SM21})
#endif(CUDPP_GENCODE_SM21)

if (CUDPP_GENCODE_SM30)
  set(GENCODE ${GENCODE} ${GENCODE_SM30})
endif(CUDPP_GENCODE_SM30)

if (CUDPP_GENCODE_SM35)
  set(GENCODE ${GENCODE} ${GENCODE_SM35})
endif(CUDPP_GENCODE_SM35)

if (CUDPP_GENCODE_SM50)
  set(GENCODE ${GENCODE} ${GENCODE_SM50})
endif(CUDPP_GENCODE_SM50)

if (CUDPP_GENCODE_SM61)
  set(GENCODE ${GENCODE} ${GENCODE_SM61})
endif(CUDPP_GENCODE_SM61)

if (CUDA_VERBOSE_PTXAS)
  set(VERBOSE_PTXAS --ptxas-options=-v)
endif (CUDA_VERBOSE_PTXAS)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  exec_program(uname ARGS -v  OUTPUT_VARIABLE DARWIN_VERSION)
  string(REGEX MATCH "[0-9]+" DARWIN_VERSION ${DARWIN_VERSION})
  # message(STATUS "Detected Darwin kernel version: ${DARWIN_VERSION}")
  if ((DARWIN_VERSION GREATER 13) OR (DARWIN_VERSION EQUAL 13))
    set(APPLE_MAVERICKS_OR_LATER 1 INTERNAL)
    set(CMAKE_CXX_FLAGS -stdlib=libstdc++)
    set(CMAKE_C_FLAGS -stdlib=libstdc++)
    if(CUDA_VERSION_MAJOR EQUAL 6 AND CUDA_VERSION_MINOR LESS 5)
      set(CUDA_6_0 1)
    endif(CUDA_VERSION_MAJOR EQUAL 6 AND CUDA_VERSION_MINOR LESS 5)
    if(CUDA_VERSION_MAJOR LESS 6 OR CUDA_6_0)
      if(NOT CUDA_NVCC_FLAGS MATCHES "ccbin")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -ccbin;/usr/bin/clang)
      endif(NOT CUDA_NVCC_FLAGS MATCHES "ccbin")
    endif(CUDA_VERSION_MAJOR LESS 6 OR CUDA_6_0)
  endif ((DARWIN_VERSION GREATER 13) OR (DARWIN_VERSION EQUAL 13))
  # current CUDA on Mavericks uses libstdc++, not libc++
  # https://github.com/cudpp/cudpp/wiki/BuildingCUDPPWithMavericks
endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

cuda_add_library(cudpp_util ${LIB_TYPE}
#  ${HFILES}
  ${CUHFILES}
  ${HFILES_PUBLIC}
  ${CUFILES}
  OPTIONS ${GENCODE} ${VERBOSE_PTXAS}
  )

install(FILES ${HFILES_PUBLIC}
  DESTINATION include
  )

install(TARGETS cudpp_util
  DESTINATION lib
  EXPORT cudpp-targets
  )

install(EXPORT cudpp-targets
  DESTINATION lib
  )
