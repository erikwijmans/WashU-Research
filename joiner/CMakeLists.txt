set(CMAKE_CXX_COMPILER g++)
cmake_minimum_required(VERSION 2.8)
project(joiner)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Ofast -g")
find_package(gflags REQUIRED)
find_package(PCL REQUIRED)
find_package( Boost COMPONENTS system filesystem REQUIRED )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIRS} )
set(globals_dir ${CMAKE_CURRENT_SOURCE_DIR}/../globals)
include_directories(${globals_dir})
file(GLOB globals
    ${globals_dir}/scan_gflags.cpp
    ${globals_dir}/scan_typedefs.cpp)

file(GLOB src
	"joiner.cpp")

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
add_executable( joiner ${src} ${globals})
target_link_libraries( joiner ${PCL_LIBRARIES} ${Boost_FILESYSTEM_LIBRARY}
                      ${Boost_SYSTEM_LIBRARY} gflags)
