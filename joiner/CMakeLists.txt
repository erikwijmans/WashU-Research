project(joiner CXX)

if(APPLE)
  set(CMAKE_CXX_COMPILER /usr/local/bin/clang++)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lc++ -lc++abi")
else()
  set(CMAKE_CXX_COMPILER g++-4.9)
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++14 -O3 -g -fopenmp")
find_package( PCL REQUIRED )
find_package( OpenCV REQUIRED )
include_directories(${globals_INCLUDE})
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

file(GLOB src
	"joiner.cpp")

add_executable( joiner ${src})
target_link_libraries( joiner ${globals_LIBS} ${OpenCV_LIBS} ${PCL_LIBRARIES})
cotire(joiner)
