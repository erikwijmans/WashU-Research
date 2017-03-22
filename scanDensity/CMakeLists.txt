cmake_minimum_required(VERSION 3.1)
project( scanDensity CXX)

find_package( OpenCV REQUIRED )
find_package( Boost REQUIRED timer thread REQUIRED )
find_package( gflags REQUIRED )
find_package( glog REQUIRED )


file( GLOB scan_SRC
    "scanDensity*.cpp"
    "3DInfo.cpp"
    "driver.cpp")
add_executable( scanDensity ${scan_SRC})
target_link_libraries( scanDensity PRIVATE etw_utils ${OpenCV_LIBS}
                      ${Boost_LIBRARIES}
                      glog)
target_include_directories(scanDensity PRIVATE ${Boost_INCLUDE_DIRS})
set_property(TARGET scanDensity PROPERTY CXX_STANDARD 14)
cotire(scanDensity)