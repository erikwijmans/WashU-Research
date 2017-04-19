project(blurrer CXX)

find_package( PCL REQUIRED )
find_package( OpenCV REQUIRED )

file(GLOB src
  "*.cpp"
)

add_executable( blurrer ${src})
target_link_libraries( blurrer PRIVATE etw_utils
                      ${PCL_LIBRARIES} ${OpenCV_LIBS})
target_include_directories(blurrer PRIVATE
                           ${PCL_INCLUDE_DIRS})
target_compile_definitions(blurrer PRIVATE ${PCL_DEFINITIONS})
cotire(blurrer)
