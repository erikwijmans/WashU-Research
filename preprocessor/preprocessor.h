#pragma once
#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H

#include <etw_utils.hpp>
#include <scan_gflags.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

typedef pcl::PointXYZRGB PointType;
typedef pcl::Normal NormalType;
typedef pcl::ReferenceFrame RFType;
typedef pcl::SHOT352 DescriptorType;

void convertToBinary(const fs::path &fileNameIn, const fs::path &,
                     std::vector<scan::PointXYZRGBA> &pointCloud);
void createPanorama(const std::vector<scan::PointXYZRGBA> &pointCloud,
                    pcl::PointCloud<NormalType>::Ptr &cloud_normals,
                    pcl::PointCloud<PointType>::Ptr &normals_points,
                    const fs::path &panoName, const fs::path &dataName);
void boundingBox(const std::vector<scan::PointXYZRGBA> &points,
                 Eigen::Vector3f &pointMin, Eigen::Vector3f &pointMax);
void createPCLPointCloud(const std::vector<scan::PointXYZRGBA> &points,
                         pcl::PointCloud<PointType>::Ptr &cloud);
void getNormals(const pcl::PointCloud<PointType>::Ptr &cloud,
                pcl::PointCloud<NormalType>::Ptr &cloud_normals,
                pcl::PointCloud<PointType>::Ptr &normals_points,
                const fs::path &outName);
void saveNormals(const pcl::PointCloud<NormalType>::Ptr &cloud_normals,
                 pcl::PointCloud<PointType>::Ptr &normals_points,
                 const fs::path &outName);

void findDoors(pcl::PointCloud<PointType>::Ptr &pointCloud,
               const Eigen::Vector3d &M1, const Eigen::Vector3d &M2,
               const Eigen::Vector3d &M3, const fs::path &outName);

#endif // PREPROCESSOR_H
