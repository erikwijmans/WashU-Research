#ifndef SCAN_DENSITY_3D_INFO_H
#define SCAN_DENSITY_3D_INFO_H

#include "scanDensity_scanDensity.h"

#include <unordered_map>
#include <FeatureVoxel.hpp>

namespace std {
  template <>
  struct hash<Eigen::Vector3i> {
    std::size_t operator()(const Eigen::Vector3i & k) const {
      size_t seed = 0;
      for (int i = 0; i < k.size(); ++i) {
        seed ^= k[i] + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      }
      return seed;
    }
  };
} // std

namespace voxel {

  class CloudAnalyzer3D {
    private:
      DensityMapsManager::PointsPtr points;
      DensityMapsManager::MatPtr R;
      BoundingBox::ConstPtr bBox;
      DensityMapsManager::FeaturePtr featureVectors;
      std::unordered_map<Eigen::Vector3i, FeatureVoxel<float>::DescripPtr> xyzToSHOT;
      std::vector<Eigen::MatrixXi> pointsPerVoxel, numTimesSeen;
      Eigen::Vector3f pointMin, pointMax;
      double voxelsPerMeter, pixelsPerMeter;
      Eigen::Vector3d zeroZeroD;
      Eigen::Vector3i zeroZero;
    public:
      typedef std::shared_ptr<voxel::CloudAnalyzer3D> Ptr;
      CloudAnalyzer3D(const DensityMapsManager::PointsPtr & points,
        const DensityMapsManager::MatPtr & R,
        const DensityMapsManager::FeaturePtr & featureVectors,
        const BoundingBox::ConstPtr & bBox);
      void run(double voxelsPerMeter, double pixelsPerMeter);
      void saveVoxelGrids(const std::vector<std::string> & pointNames,
        const std::vector<std::string> & freeNames,
        const std::string & metaData);
  };

} // voxel

#endif // SCAN_DENSITY_3D_INFO_H
