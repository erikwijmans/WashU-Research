set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Ofast -g -fopenmp")
cmake_minimum_required(VERSION 2.8)
project( scanDensity )
find_package( OpenCV REQUIRED )
find_package(gflags REQUIRED)
file( GLOB scan_SRC
    "scanDensity*.cpp"
    "3DInfo.cpp"
    "driver.cpp")
add_executable( scanDensity ${scan_SRC})
target_link_libraries( scanDensity ${OpenCV_LIBS} gflags)

project(preprocessor)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Ofast -fopenmp -g")
find_package(gflags REQUIRED)
find_package( OpenCV REQUIRED )
add_executable( preprocessor preprocessor.cpp )
target_link_libraries( preprocessor ${OpenCV_LIBS} gflags)