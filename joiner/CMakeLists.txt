project(joiner CXX)

find_package( PCL REQUIRED )


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")

file(GLOB src
	"joiner.cpp")

add_executable( joiner ${src})
target_link_libraries( joiner PRIVATE etw_utils
                      ${PCL_LIBRARIES})
target_include_directories(joiner PRIVATE ${PCL_INCLUDE_DIRS})
target_compile_definitions(joiner PRIVATE ${PCL_DEFINITIONS})
cotire(joiner)
