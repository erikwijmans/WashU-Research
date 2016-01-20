#ifndef SCAN_DENSITY_3D_INFO_H
#define SCAN_DENSITY_3D_INFO_H

#include "scanDensity_scanDensity.h"

#define NUM_ROTS 4

namespace voxel {

  void saveVoxelGrid(std::vector<Eigen::MatrixXi> & grid,
    const std::string & outName);

  void createVoxelGrids(const std::vector<Eigen::Vector3f> & points,
    const float * pointMin, const float * pointMax,  
    const std::string & rotationFile, const std::string & scanNumber);

  void pointBased(const std::vector<Eigen::Vector3f> & points,
    const Eigen::Vector3d & pointMin, const Eigen::Vector3d & pointMax,
    const Eigen::Matrix3d & rot, const std::string & scanNumber,
    const int rotNumber);

  void freeSpace(const std::vector<Eigen::Vector3f> & points,
    const Eigen::Vector3d & pointMin, const Eigen::Vector3d & pointMax,
    const Eigen::Matrix3d & rot, const std::string & scanNumber,
    const int rotNumber);

} // 3D




#endif // SCAN_DENSITY_3D_INFO_H
