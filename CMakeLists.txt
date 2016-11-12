cmake_minimum_required(VERSION 3.1)

execute_process(COMMAND "git submodule update --init --recursive"
                WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
set (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cotire/CMake")
project(Laser_Scan_Placement)

include(cotire)
add_subdirectory(globals)
add_subdirectory(preprocessor)
add_subdirectory(scanDensity)
add_subdirectory(placeScan)
add_subdirectory(joiner)
add_subdirectory(k4pcs)