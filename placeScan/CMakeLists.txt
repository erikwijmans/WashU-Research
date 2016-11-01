project( placeScan CXX)
if(APPLE)
  set(CMAKE_CXX_COMPILER /usr/local/bin/clang++)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lc++ -lc++abi")
else()
  set(CMAKE_CXX_COMPILER g++)
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++14 -O3 -g -fopenmp")
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS timer REQUIRED)

find_package( gflags REQUIRED )
include_directories( ${Boost_INCLUDE_DIRS} )
include_directories(${globals_INCLUDE})


file( GLOB place_SRC
   "place*.cpp"
   "multiLabeling.cpp"
   "panoramaMatcher.cpp"
   "highOrder.cpp"
   "doorDetector.cpp")

add_executable( placeScan ${place_SRC})
target_link_libraries( placeScan ${globals_LIBS} ${OpenCV_LIBS}
                      ${Boost_TIMER_LIBRARIES}
                      ${OpenGM_LIBS} gflags)
cotire(placeScan)
