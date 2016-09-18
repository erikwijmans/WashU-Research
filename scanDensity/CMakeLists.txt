cmake_minimum_required(VERSION 3.1)
project( scanDensity CXX)
if(APPLE)
  set(CMAKE_CXX_COMPILER /usr/local/bin/clang++)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lc++ -lc++abi")
else()
  set(CMAKE_CXX_COMPILER g++)
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++14 -O3 -g -fopenmp")
find_package( OpenCV REQUIRED )
find_package( Boost REQUIRED )
find_package( gflags REQUIRED )
include_directories( ${Boost_INCLUDE_DIRS} )
include_directories(${globals_INCLUDE})

file( GLOB scan_SRC
    "scanDensity*.cpp"
    "3DInfo.cpp"
    "driver.cpp")
add_executable( scanDensity ${scan_SRC})
target_link_libraries( scanDensity ${globals_LIBS} ${OpenCV_LIBS} gflags)
cotire(scanDensity)