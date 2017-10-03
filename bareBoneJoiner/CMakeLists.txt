cmake_minimum_required(VERSION 3.1)
project(BareBoneJoiner)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_FLAGS "-std=c++14 -O2 -g")


find_package( PCL REQUIRED )
find_package( gflags REQUIRED )
find_package( fmt REQUIRED )
find_package( Boost COMPONENTS system filesystem REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")

file(GLOB src
	"bare_bones_joiner.cpp")

add_executable( bare_bones_joiner ${src})
target_link_libraries( bare_bones_joiner PRIVATE
                      ${PCL_LIBRARIES} gflags fmt
                      ${Boost_FILESYSTEM_LIBRARIES})
target_include_directories(bare_bones_joiner PRIVATE ${PCL_INCLUDE_DIRS}
                                ${Boost_INCLUDE_DIRS})
