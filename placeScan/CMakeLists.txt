project( placeScan CXX)
find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS timer REQUIRED)
find_package( gflags REQUIRED )
find_package( glog REQUIRED )
include_directories( ${Boost_INCLUDE_DIRS} )
include_directories(${globals_INCLUDE})

file( GLOB place_SRC
   "place*.cpp"
   "multiLabeling.cpp"
   "panoramaMatcher.cpp"
   "highOrder.cpp"
   "doorDetector.cpp")

add_executable( placeScan ${place_SRC})
target_link_libraries( placeScan PRIVATE etw_utils
                      ${OpenCV_LIBS}
                      ${Boost_TIMER_LIBRARIES} gflags
                      glog)
target_include_directories(placeScan PRIVATE ${Boost_INCLUDE_DIRS})
cotire(placeScan)
