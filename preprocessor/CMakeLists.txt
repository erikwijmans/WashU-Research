project(preprocessor CXX)

find_package( PCL REQUIRED )

file(GLOB src
	"preprocessor.cpp"
	"getRotations.cpp")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")

add_executable( preprocessor ${src})
target_link_libraries( preprocessor PRIVATE etw_utils
                      ${PCL_LIBRARIES})
target_include_directories(preprocessor PRIVATE
                           ${PCL_INCLUDE_DIRS})
target_compile_definitions(preprocessor PRIVATE ${PCL_DEFINITIONS})
cotire(preprocessor)
