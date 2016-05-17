if(APPLE)
  set(CMAKE_CXX_COMPILER clang++)
else()
  set(CMAKE_CXX_COMPILER g++-4.9)
endif()
cmake_minimum_required(VERSION 2.8)
project( scanDensity )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++14 -O3 -g -fopenmp")
find_package( OpenCV REQUIRED )
find_package(gflags REQUIRED)
find_package( Boost REQUIRED )
include_directories( ${Boost_INCLUDE_DIRS} )

set(globals_dir ${CMAKE_CURRENT_SOURCE_DIR}/../globals)
include_directories(${globals_dir})
file(GLOB scan_globals
	${globals_dir}/scan_gflags.cpp)

file( GLOB scan_SRC
    "scanDensity*.cpp"
    "3DInfo.cpp"
    "driver.cpp")
add_executable( scanDensity ${scan_SRC} ${scan_globals})
target_link_libraries( scanDensity ${OpenCV_LIBS} gflags)