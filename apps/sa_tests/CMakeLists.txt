###############################################################################
#
# Build script for project
#
###############################################################################
cmake_minimum_required(VERSION 2.8)

find_package(CUDA REQUIRED)

# Add include/lib paths for local cmake builds
include_directories(${CMAKE_SOURCE_DIR}/../../include)
link_directories(${CMAKE_SOURCE_DIR}/../../lib/)

set(CUFILES
    cuda_util.cu
    cuda_wrapper.cu
  )


set(HFILES
    run_sa.h
  )

#source_group("CUDA Source Files" FILES ${CUFILES})

include_directories(../common/include)

cuda_add_executable(run_sa ${HFILES})

target_link_libraries(run_sa
  cudpp
  )
