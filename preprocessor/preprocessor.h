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

void convertToBinary(const std::string &fileNameIn, const std::string &,
                     std::vector<scan::PointXYZRGBA> &pointCloud);
void createPanorama(const std::vector<scan::PointXYZRGBA> &pointCloud,
                    pcl::PointCloud<NormalType>::Ptr &cloud_normals,
                    pcl::PointCloud<PointType>::Ptr &normals_points,
                    const std::string &panoName, const std::string &dataName);
void boundingBox(const std::vector<scan::PointXYZRGBA> &points,
                 Eigen::Vector3f &pointMin, Eigen::Vector3f &pointMax);
void createPCLPointCloud(const std::vector<scan::PointXYZRGBA> &points,
                         pcl::PointCloud<PointType>::Ptr &cloud);
void getNormals(const pcl::PointCloud<PointType>::Ptr &cloud,
                pcl::PointCloud<NormalType>::Ptr &cloud_normals,
                pcl::PointCloud<PointType>::Ptr &normals_points,
                const std::string &outName);
void saveNormals(const pcl::PointCloud<NormalType>::Ptr &cloud_normals,
                 pcl::PointCloud<PointType>::Ptr &normals_points,
                 const std::string &outName);

#endif // PREPROCESSOR_H
