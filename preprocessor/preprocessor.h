#pragma once
#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H

#include <scan_gflags.h>
#include <scan_typedefs.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

typedef pcl::PointXYZRGB PointType;
typedef pcl::Normal NormalType;
typedef pcl::ReferenceFrame RFType;
typedef pcl::SHOT352 DescriptorType;

#endif // PREPROCESSOR_H
