cmake_minimum_required(VERSION 3.1)
find_package( OpenCV REQUIRED )
find_package( gflags REQUIRED )
find_package( fmt REQUIRED )
find_package( Boost COMPONENTS System Filesystem REQUIRED)
find_package( glog REQUIRED )

set_property(TARGET fmt PROPERTY INTERFACE_COMPILE_OPTIONS "")

set(utils_SRC
  scan_gflags.cpp
  etw_utils.cpp
  RGBD_utils.cpp)

add_library(etw_utils ${utils_SRC})
target_link_libraries(etw_utils PUBLIC ${OpenCV_LIBS} ${Boost_SYSTEM_LIBRARIES}
                      ${Boost_FILESYSTEM_LIBRARIES}
                      gflags fmt glog unwind)
target_include_directories(etw_utils PUBLIC ${Boost_INCLUDE_DIRS}
                           ${CMAKE_CURRENT_SOURCE_DIR})
cotire(etw_utils)