set(CMAKE_CXX_COMPILER g++-4.9)
cmake_minimum_required(VERSION 2.8)
find_package(gflags REQUIRED)
project( getRotations )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Ofast -fopenmp")

set(globals_dir ${CMAKE_CURRENT_SOURCE_DIR}/../globals)
include_directories(${globals_dir})
file(GLOB scan_globals 
	${globals_dir}/scan_gflags.cpp
	)

add_executable( getRotations getRotations.cpp ${scan_globals})
target_link_libraries(getRotations gflags)