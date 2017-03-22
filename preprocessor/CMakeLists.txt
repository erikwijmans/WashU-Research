project(preprocessor CXX)

find_package( PCL REQUIRED )
find_package( OpenCV REQUIRED )
find_package( Boost REQUIRED )
find_package( fmt REQUIRED)


file(GLOB src
	"preprocessor.cpp"
	"getRotations.cpp")

add_executable( preprocessor ${src})
target_link_libraries( preprocessor PRIVATE etw_utils
                      ${OpenCV_LIBS}
                      ${PCL_LIBRARIES}
                      fmt)
target_include_directories(preprocessor PRIVATE
                           ${PCL_INCLUDE_DIRS}
                           ${Boost_INCLUDE_DIRS})
target_compile_definitions(preprocessor PRIVATE ${PCL_DEFINITIONS})
set_property(TARGET preprocessor PROPERTY CXX_STANDARD 14)
cotire(preprocessor)
