cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_COMPILER g++-4.9)
set(Gurobi /opt/gurobi650/linux64/lib/libgurobi_c++.a 
	/opt/gurobi650/linux64/lib/libgurobi65.so)
project( placeScan )
find_package( OpenCV REQUIRED )
find_package( gflags REQUIRED )
file( GLOB place_SRC
   "place*.cpp" 
   "multiLabeling.cpp"
   "panoramaMatcher.cpp")
include_directories(/opt/gurobi650/linux64/include/)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -std=gnu++14 -Ofast -g")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lgurobi_c++ -lgurobi65 -lm")

set(globals_dir ${CMAKE_CURRENT_SOURCE_DIR}/../globals)
include_directories(${globals_dir})
file(GLOB scan_globals 
	${globals_dir}/scan_gflags.cpp
	${globals_dir}/FeatureVoxel.cpp)

add_executable( placeScan ${place_SRC} ${scan_globals})
target_link_libraries( placeScan ${OpenCV_LIBS} ${Gurobi} gflags)