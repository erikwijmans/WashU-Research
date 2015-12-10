cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_COMPILER g++)
project( placeScan )
find_package( OpenCV REQUIRED )
find_package( gflags REQUIRED )
file( GLOB place_SRC
   "*.cpp"
   "*.h")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -std=gnu++14 -g -Ofast")
add_executable( placeScan ${place_SRC})
target_link_libraries( placeScan ${OpenCV_LIBS} gflags)