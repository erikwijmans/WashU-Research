find_package( OpenCV REQUIRED )
find_package( gflags REQUIRED )
find_package( fmt REQUIRED )
find_package( Boost COMPONENTS system filesystem REQUIRED )
find_package( glog REQUIRED )

set(utils_SRC
  scan_gflags.cpp
  etw_utils.cpp)


add_library(etw_utils ${utils_SRC})
target_link_libraries(etw_utils PUBLIC ${OpenCV_LIBS} ${Boost_SYSTEM_LIBRARIES}
                      ${Boost_FILESYSTEM_LIBRARIES} gflags fmt glog)
target_include_directories(etw_utils PUBLIC ${Boost_INCLUDE_DIRS}
                           ${CMAKE_CURRENT_SOURCE_DIR})
set_property(TARGET etw_utils PROPERTY CXX_STANDARD 14)

cotire(etw_utils)