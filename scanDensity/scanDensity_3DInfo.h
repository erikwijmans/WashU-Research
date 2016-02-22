#ifndef SCAN_DENSITY_3D_INFO_H
#define SCAN_DENSITY_3D_INFO_H

#include "scanDensity_scanDensity.h"

#define NUM_ROTS 4

const double voxelsPerMeter = 20.0;

namespace Eigen {
  typedef Matrix<char, Dynamic, Dynamic> MatrixXb;
}


namespace voxel {

  typedef struct {
    Eigen::Vector3i zz;
    int x, y, z;
  } metaData;

  void analyzeScan3D(const std::string & fileName,
  const std::string & rotationFile);

  void saveVoxelGrid(std::vector<Eigen::MatrixXi> & grid,
    const std::vector<Eigen::Matrix3d> & R,
    const Eigen::Vector3d & zeroZero,
    const std::string & scanNumber, const std::string & type);


  void createVoxelGrids(const std::vector<Eigen::Vector3f> & points,
    const std::string & rotationFile, const std::string & scanNumber);

  void pointBased(const std::vector<Eigen::Vector3f> & points,
    const std::vector<Eigen::Matrix3d> & R,
    float * pointMin, float * pointMax,
    const std::string & scanNumber);

  void freeSpace(const std::vector<Eigen::Vector3f> & points,
    const float * pointMin, const float * pointMax,
    const std::string & scanNumber, const int rotNumber);

  void createBoundingBox(float * pointMin, float * pointMax,
    const std::vector<Eigen::Vector3f> & points);

} // 3D




#endif // SCAN_DENSITY_3D_INFO_H
