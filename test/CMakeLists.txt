set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Ofast -fopenmp")
cmake_minimum_required(VERSION 2.8)
project( test )
find_package( OpenCV REQUIRED )
find_package(gflags REQUIRED)
file(GLOB test_SRC
    "*.h"
    "*.cpp"
    "*.hpp"
)
add_executable( test ${test_SRC})
target_link_libraries( test ${OpenCV_LIBS} gflags)