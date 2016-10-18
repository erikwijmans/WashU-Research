#ifndef GET_ROTATIONS_H
#define GET_ROTATIONS_H

#include "preprocessor.h"

void getRotations(const pcl::PointCloud<NormalType>::Ptr &cloud_normals,
                  const std::string &outName, Eigen::Vector3d &M1,
                  Eigen::Vector3d &M2, Eigen::Vector3d &M3);

#endif // GET_ROTATIONS_H
