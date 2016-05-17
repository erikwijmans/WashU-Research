set(CMAKE_CXX_COMPILER clang++)
cmake_minimum_required(VERSION 2.8)
project(test)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Ofast -fopenmp -g")
find_package( Boost COMPONENTS system filesystem REQUIRED )
find_package(gflags REQUIRED)
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIRS} )

add_executable(test scan_gflags.cpp)
target_link_libraries( test
                      ${Boost_FILESYSTEM_LIBRARY}
                      ${Boost_SYSTEM_LIBRARY} gflags)