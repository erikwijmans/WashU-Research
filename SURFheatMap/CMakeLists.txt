cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_COMPILER g++)
project( heatMap )
find_package( OpenCV REQUIRED )
file( GLOB src
   "heatMap.cpp")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -std=gnu++14 -Ofast -g")

add_executable( heatMap ${src})
target_link_libraries( heatMap ${OpenCV_LIBS})