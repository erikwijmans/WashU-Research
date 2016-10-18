#ifndef SCAN_DENSITY_3D_INFO_H
#define SCAN_DENSITY_3D_INFO_H

#include "scanDensity_scanDensity.h"

#include <FeatureVoxel.hpp>
#include <unordered_map>

namespace voxel {

class CloudAnalyzer3D {
private:
  DensityMapsManager::PointsPtr points;
  DensityMapsManager::MatPtr R;
  BoundingBox::ConstPtr bBox;
  DensityMapsManager::FeaturePtr featureVectors;
  std::unordered_map<Eigen::Vector3i, FeatureVoxel<float>::DescripPtr>
      xyzToSHOT;
  std::vector<Eigen::MatrixXi> pointsPerVoxel, numTimesSeen;
  Eigen::Vector3f pointMin, pointMax;
  double voxelsPerMeter, pixelsPerMeter;
  Eigen::Vector3d zeroZeroD;
  Eigen::Vector3i zeroZero;

public:
  typedef std::shared_ptr<voxel::CloudAnalyzer3D> Ptr;
  CloudAnalyzer3D(const DensityMapsManager::PointsPtr &points,
                  const DensityMapsManager::MatPtr &R,
                  const BoundingBox::ConstPtr &bBox);
  void run(double voxelsPerMeter, double pixelsPerMeter);
  void saveVoxelGrids(const std::vector<std::string> &pointNames,
                      const std::vector<std::string> &freeNames,
                      const std::string &metaData);
};

} // voxel

#endif // SCAN_DENSITY_3D_INFO_H
