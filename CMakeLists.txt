cmake_minimum_required(VERSION 3.1)
set (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cotire/CMake")
project(Laser_Scan_Placement)

if(APPLE)
  set(CMAKE_CXX_COMPILER /usr/local/bin/clang++)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lc++ -lc++abi")
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)

include(cotire)
set(CMAKE_CXX_FLAGS "-std=gnu++14 -O2 -g -fopenmp")


add_subdirectory(utils)
add_subdirectory(preprocessor)
add_subdirectory(scanDensity)
add_subdirectory(placeScan)
add_subdirectory(joiner)