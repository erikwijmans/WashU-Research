set(CMAKE_CXX_COMPILER g++)
cmake_minimum_required(VERSION 2.8)
project( scanDensity )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Ofast -g")
find_package( OpenCV REQUIRED )
find_package(gflags REQUIRED)
include_directories(/home/erik/Projects/c++/globals)
file( GLOB scan_SRC
    "scanDensity*.cpp"
    "3DInfo.cpp"
    "driver.cpp")
add_executable( scanDensity ${scan_SRC})
target_link_libraries( scanDensity ${OpenCV_LIBS} gflags)