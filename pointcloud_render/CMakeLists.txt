cmake_minimum_required (VERSION 2.6 FATAL_ERROR)

project      (renderer)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# set(CMAKE_AUTOMOC ON)

find_package (Qt5 COMPONENTS Core Gui OpenGL OpenGLExtensions Widgets REQUIRED)
find_package (PCL 1.7.1 REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++14 -O3 -g")

include_directories (${globals_INCLUDE})

include_directories (${PCL_INCLUDE_DIRS})
link_directories    (${PCL_LIBRARY_DIRS})
add_definitions     (${PCL_DEFINITIONS})

include_directories (${Qt5Core_INCLUDE_DIRS})
link_directories    (${Qt5Core_LIBRARY_DIRS})
add_definitions     (${Qt5Core_DEFINITIONS})

include_directories (${Qt5Gui_INCLUDE_DIRS})
link_directories    (${Qt5Gui_LIBRARY_DIRS})
add_definitions     (${Qt5Gui_DEFINITIONS})

include_directories (${Qt5OpenGL_INCLUDE_DIRS})
link_directories    (${Qt5OpenGL_LIBRARY_DIRS})
add_definitions     (${Qt5OpenGL_DEFINITIONS})

include_directories (${Qt5OpenGLExtensions_INCLUDE_DIRS})
link_directories    (${Qt5OpenGLExtensions_LIBRARY_DIRS})
add_definitions     (${Qt5OpenGLExtensions_DEFINITIONS})

include_directories (${Qt5Widgets_INCLUDE_DIRS})
link_directories    (${Qt5Widgets_LIBRARY_DIRS})
add_definitions     (${Qt5Widgets_DEFINITIONS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Qt5Widgets_EXECUTABLE_COMPILE_FLAGS}")

set  (project_SOURCES main.cpp widget.cpp)
set  (project_HEADERS widget.h)
set  (project_FORMS   widget.ui)

QT5_WRAP_CPP (project_HEADERS_MOC   ${project_HEADERS})
QT5_WRAP_UI  (project_FORMS_HEADERS ${project_FORMS})


ADD_EXECUTABLE  (pointcloud_renderer ${project_SOURCES}
                                ${project_FORMS_HEADERS}
                                ${project_HEADERS_MOC})

TARGET_LINK_LIBRARIES (pointcloud_renderer ${PCL_LIBRARIES}
                       ${Qt5Widgets_LIBRARIES}
                       ${Qt5Core_LIBRARIES}
                       ${Qt5Gui_LIBRARIES}
                       ${Qt5OpenGL_LIBRARIES}
                       ${Qt5OpenGLExtensions_LIBRARIES}
                       ${globals_LIBS}
                       ${OpenCV_LIBS})
