cmake_minimum_required(VERSION 2.8)
if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  set(CMAKE_CXX_COMPILER g++-4.9)
else
  set(CMAKE_CXX_COMPILER clang++)
endif
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")
project( placeScan )
find_package( OpenCV REQUIRED )
find_package( gflags REQUIRED )
find_package( GUROBI REQUIRED )
include_directories(${GUROBI_INCLUDE_DIRS})
file( GLOB place_SRC
   "place*.cpp"
   "multiLabeling.cpp"
   "panoramaMatcher.cpp"
   "highOrder.cpp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -std=gnu++14 -Ofast -g")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${GUROBI_")
set(globals_dir ${CMAKE_CURRENT_SOURCE_DIR}/../globals)
include_directories(${globals_dir})
file(GLOB scan_globals
  ${globals_dir}/scan_gflags.cpp)

add_executable( placeScan ${place_SRC} ${scan_globals})
target_link_libraries( placeScan ${OpenCV_LIBS} ${GUROBI_LIBS} gflags)