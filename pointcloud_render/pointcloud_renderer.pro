#-------------------------------------------------
#
# Project created by QtCreator 2016-11-16T13:40:46
#
#-------------------------------------------------

QT       += core gui opengl

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = pointcloud_renderer
TEMPLATE = app

LIBS+= -lGLU


SOURCES += main.cpp\
        widget.cpp

HEADERS  += widget.h

FORMS    += widget.ui
