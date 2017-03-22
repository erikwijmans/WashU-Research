project(joiner CXX)

find_package( PCL REQUIRED )
find_package( OpenCV REQUIRED )

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

file(GLOB src
	"joiner.cpp")

add_executable( joiner ${src})
target_link_libraries( joiner PRIVATE etw_utils
                      ${OpenCV_LIBS} ${PCL_LIBRARIES})
target_include_directories(joiner PRIVATE ${PCL_INCLUDE_DIRS})
set_property(TARGET joiner PROPERTY CXX_STANDARD 14)
cotire(joiner)
