if(APPLE)
  set(CMAKE_CXX_COMPILER /usr/local/bin/clang++)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lc++ -lc++abi")
else()
  set(CMAKE_CXX_COMPILER g++)
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++14 -O3 -g -fopenmp")
find_package( OpenCV REQUIRED )
find_package( gflags REQUIRED )
find_package( Boost COMPONENTS system filesystem REQUIRED )
include_directories( ${Boost_INCLUDE_DIRS} )

set(globals_SRC
  scan_gflags.cpp
  scan_typedefs.cpp)

add_library(globals_lib ${globals_SRC})
target_link_libraries(globals_lib ${OpenCV_LIBS} ${Boost_SYSTEM_LIBRARIES}
                      ${Boost_FILESYSTEM_LIBRARIES} gflags)
cotire(globals_lib)

set(globals_LIBS globals_lib ${OpenCV_LIBS} ${Boost_SYSTEM_LIBRARIES}
                      ${Boost_FILESYSTEM_LIBRARIES} gflags)
set(globals_LIBS ${globals_LIBS} PARENT_SCOPE)

set(globals_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR} ${Boost_INCLUDE_DIRS})
set(globals_INCLUDE ${globals_INCLUDE} PARENT_SCOPE)