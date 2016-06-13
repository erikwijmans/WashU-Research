cmake_minimum_required(VERSION 2.8)
if (APPLE)
  set(CMAKE_CXX_COMPILER clang++)
else()
  set(CMAKE_CXX_COMPILER g++-4.9)
endif()
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
project( placeScan )
find_package( OpenCV REQUIRED )
find_package( gflags REQUIRED )
find_package( GUROBI REQUIRED )
find_package( Boost COMPONENTS system filesystem REQUIRED )
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${GUROBI_INCLUDE_DIRS})
file( GLOB place_SRC
   "place*.cpp"
   "multiLabeling.cpp"
   "panoramaMatcher.cpp"
   "highOrder.cpp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -std=gnu++14 -g -O3")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lgurobi_c++ -lgurobi65 -lm -lpthread")
set(globals_dir ${CMAKE_CURRENT_SOURCE_DIR}/../globals)
include_directories(${globals_dir})
file(GLOB scan_globals
  ${globals_dir}/scan_gflags.cpp
  ${globals_dir}/scan_typedefs.cpp)

add_executable( placeScan ${place_SRC} ${scan_globals})
target_link_libraries( placeScan ${OpenCV_LIBS}
                      ${Boost_SYSTEM_LIBRARIES}
                      ${Boost_FILESYSTEM_LIBRARIES}
                      ${GUROBI_LIBS}
                      gflags)