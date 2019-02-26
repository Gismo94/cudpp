###############################################################################
#
# Build script for project
#
###############################################################################
cmake_minimum_required(VERSION 2.8)

find_package(CUDA REQUIRED)

# Add include/lib paths for local cmake builds
include_directories(${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/cudpp
    ${CMAKE_SOURCE_DIR}/src/util
    ${CMAKE_SOURCE_DIR}/apps/cudpp_testrig
)
link_directories(${CMAKE_SOURCE_DIR}/../../lib/)

set(CCFILES
    run_sa.cpp
  )


include_directories(../common/include)
link_directories(${CMAKE_SOURCE_DIR}/../../lib/)

cuda_add_executable(run_sa ${CCFILES})

target_link_libraries(run_sa
  cudpp
  cudpp_util
  )
